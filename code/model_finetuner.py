import argparse, os, json, logging
from pathlib import Path
from typing import List, Dict, Any
from types import SimpleNamespace

import torch, datasets, evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# task registry --------------------------------------------------------------
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {}

def register(name):
    def _wrap(fn):
        TASKS[name] = fn
        return fn
    return _wrap


def _filter_langs(requested: List[str], available: List[str], task_name: str):
    # Helper: keep only languages that exist in the dataset
    ok = [lg for lg in requested if lg in available]
    miss = [lg for lg in requested if lg not in available]
    logger.info("%s | requested=%s  available=%s", task_name, requested, available)
    if miss:
        logger.warning("%s | languages not found and skipped: %s", task_name, miss)
    if not ok:
        raise ValueError(f"{task_name}: none of the requested languages found!")
    return ok


# ---------- 1) XNLI ---------------------------------------------------------
@register("xnli")
def setup_xnli(tokenizer, lang_codes, args):
    full = datasets.load_dataset("xnli")
    ok = _filter_langs(lang_codes, list(full["train"]["language"]), "XNLI")
    ds = full.filter(lambda ex: ex["language"] in ok)
    num_labels = 3
    label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def preprocess(ex):
        return tokenizer(
            ex["premise"], ex["hypothesis"],
            truncation=True, max_length=args.max_len)
    ds = ds.map(preprocess, batched=True)
    metric = evaluate.load("accuracy")

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=num_labels, label2id=label2id
    )
    return ds, model, compute

# ---------- 2) MARC ---------------------------------------------------------
@register("marc")
def setup_marc(tokenizer, lang_codes, args):
    okay = _filter_langs(lang_codes, list(datasets.get_dataset_config_names("amazon_reviews_multi")), "MARC")
    ds = datasets.concatenate_datasets([datasets.load_dataset("amazon_reviews_multi", lg)["train"] for lg in okay])

    num_labels = 5
    def preprocess(ex):
        return tokenizer(ex["review_body"], truncation=True, max_length=args.max_len)
    ds = ds.map(preprocess, batched=True)
    metric = evaluate.load("f1", average="macro")
    def compute(pred):
        logits, labels = pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=num_labels
    )
    return ds, model, compute

# ---------- 3) WikiANN NER --------------------------------------------------
@register("wikiann")
def setup_ner(tokenizer, lang_codes, args):
    okay = _filter_langs(lang_codes, list(datasets.get_dataset_config_names("wikiann")), "WikiANN")
    ds = datasets.concatenate_datasets([datasets.load_dataset("wikiann", lg)["train"] for lg in okay])

    label_list = ds.features["ner_tags"].feature.names
    num_labels = len(label_list)

    def tokenize_and_align(ex):
        tok = tokenizer(ex["tokens"], is_split_into_words=True,
                        truncation=True, max_length=args.max_len)
        tok["labels"] = ex["ner_tags"]
        return tok
    ds = ds.map(tokenize_and_align, batched=True)
    seqeval = evaluate.load("seqeval")
    def compute(p):
        preds, labels = p
        preds = preds.argmax(-1)
        true, pred = [], []
        for l, p_ in zip(labels, preds):
            true.append([label_list[i] for i in l])
            pred.append([label_list[i] for i in p_])
        res = seqeval.compute(predictions=pred, references=true)
        return {"f1": res["overall_f1"]}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path, num_labels=num_labels
    )
    return ds, model, compute

# ---------- 4) MLQA (extractive QA) ----------------------------------------
@register("mlqa")
def setup_mlqa(tokenizer, lang_codes, args):
    okay = _filter_langs(lang_codes, list(datasets.get_dataset_config_names("mlqa")), "MLQA")
    ds   = datasets.concatenate_datasets([datasets.load_dataset("mlqa", lg)["train"] for lg in okay])


    def prep(ex):
        tok = tokenizer(
            ex["context"], ex["question"],
            truncation="only_second", max_length=args.max_len,
            stride=64, return_offsets_mapping=True
        )
        return tok
    ds = ds.map(prep, batched=True)
    metric = evaluate.load("squad")
    def compute(p): return metric.compute(
        predictions=[{"id":i,"prediction_text":""} for i in range(len(p[0]))],
        references=[{"id":i,"answers":""} for i in range(len(p[1]))]
    )  # placeholder; integrate your post‑processing if you need exact scores
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    return ds, model, compute

# ---------- 5) PAWS‑X -------------------------------------------------------
@register("pawsx")
def setup_pawsx(tokenizer, lang_codes, args):
    okay = _filter_langs(lang_codes, list(datasets.get_dataset_config_names("paws-x")), "PAWS-X")
    ds   = datasets.concatenate_datasets([datasets.load_dataset("paws-x", lg)["train"] for lg in okay])
    num_labels = 2
    def preprocess(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"], truncation=True, max_length=args.max_len)
    ds = ds.map(preprocess, batched=True)
    metric = evaluate.load("accuracy")
    def compute(p):
        logits, labels = p
        return metric.compute(predictions=logits.argmax(-1), references=labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)
    return ds, model, compute

# ---------------------------------------------------------------------------
# programmatic entry‑point (used by run_experiment.py)
# ---------------------------------------------------------------------------
def finetune_from_config(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    entry = {
        'name':          'tok_x',
        'model_path':    '…',
        'tokenizer_path':'…',
        'tasks':        ['xnli','wikiann'],
        'hyper':        {'epochs':3,'batch_size':16,'lr':3e-5},
        'output_dir':   'results/…'
    }
    Returns { task -> metrics_json_path }.
    """

    args = SimpleNamespace(
        model_path   = entry["model_path"],
        tokenizer_path = entry["tokenizer_path"],
        task         = entry["tasks"],
        epochs       = entry["hyper"].get("epochs", 3),
        batch_size   = entry["hyper"].get("batch_size", 16),
        lr           = entry["hyper"].get("lr", 3e-5),
        max_len      = entry["hyper"].get("max_len", 256),
        output_dir   = entry["output_dir"],
    )
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    lang_codes = json.load(open("code/language_selection.py"))["languages"]

    metrics_map = {}
    for task in args.task:
        ds, model, compute = TASKS[task](tok, lang_codes, args)
        trainer = Trainer(
            model,
            TrainingArguments(
                output_dir   = os.path.join(args.output_dir, task),
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                evaluation_strategy="epoch",
                report_to="none",
                fp16=torch.cuda.is_available(),
            ),
            train_dataset = ds,
            eval_dataset  = ds,
            tokenizer     = tok,
            data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8),
            compute_metrics = compute,
        )
        trainer.train()
        metrics_path = os.path.join(args.output_dir, f"{task}_metrics.json")
        json.dump(trainer.evaluate(), open(metrics_path, "w"), indent=2)
        metrics_map[task] = metrics_path
    return metrics_map


# ---------------------------------------------------------------------------
# main driver ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--task", nargs="+", required=True,
                        choices=list(TASKS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    lang_codes = json.load(open("code/language_selection.py", "r"))["languages"]

    for task_name in args.task:
        logger.info("=== %s ===", task_name)
        ds, model, compute_metrics = TASKS[task_name](tok, lang_codes, args)

        #TODO: check collator
        collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
        training_args = TrainingArguments(
            output_dir = os.path.join(args.output_dir, task_name),
            evaluation_strategy = "epoch",
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size  = args.batch_size,
            num_train_epochs = args.epochs,
            learning_rate=args.lr,
            fp16=torch.cuda.is_available(),
            logging_steps = 50,
            save_strategy="no",
            report_to="none",
        )
        trainer = Trainer(
            model          = model,
            args           = training_args,
            train_dataset  = ds,
            eval_dataset   = ds,   # quick dev split; replace with validation
            tokenizer      = tok,
            data_collator  = collator,
            compute_metrics= compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        json.dump(metrics, open(f"{args.output_dir}/{task_name}_metrics.json","w"), indent=2)
        logger.info("%s metrics: %s" % (task_name, metrics))

if __name__ == "__main__": main()

