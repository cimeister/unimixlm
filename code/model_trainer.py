# model_trainer.py

import os
import logging
import json
import torch
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import Dataset, concatenate_datasets
from code.utils import fix_input_ids, nonempty, set_special_tokens

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized")
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class TokenizedDatasetLoader:
    """
    Loader for tokenized datasets in JSONL format:
      each line is {"text": "...", "token_ids": [...]}.
    """
    def __init__(self, tokenized_files):
        self.tokenized_files = tokenized_files  # e.g. {lang_code: {split: path}}
    
    def load_dataset(self, lang_code, split):
        if lang_code not in self.tokenized_files or split not in self.tokenized_files[lang_code]:
            logger.error(f"No tokenized file found for {lang_code} {split}")
            return None
        
        file_path = self.tokenized_files[lang_code][split]
        if not os.path.isfile(file_path):
            logger.error(f"File not found at {file_path}")
            return None
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append({
                        'text': item['text'],
                        'input_ids': item['token_ids']
                    })
            dataset = Dataset.from_list(data)
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {str(e)}")
            return None
    
    def load_combined_dataset(self, split):
        """
        Combine all languages' datasets for this split.
        """
        datasets = []
        for lang_code in self.tokenized_files.keys():
            ds = self.load_dataset(lang_code, split)
            if ds is not None:
                datasets.append(ds)
        if not datasets:
            logger.error(f"No datasets found for split {split}")
            return None
        
        return concatenate_datasets(datasets)


class CustomDataCollator:
    """
    Collator that pads input_ids + labels to the same length in a batch.
    """
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        for f in features:
            cur_len = len(f["input_ids"])
            pad_needed = max_len - cur_len
            if "labels" not in f:
                f["labels"] = list(f["input_ids"])
            if pad_needed > 0:
                f["input_ids"] += [self.pad_id]*pad_needed
                f["labels"]    += [self.pad_id]*pad_needed
        batch = {}
        for k in features[0].keys():
            batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
        max_id_in_batch = batch["input_ids"].max().item()
        #logger.info(f"Max token ID in this batch: {max_id_in_batch}")
        return batch


class CustomCausalLMModel:
    """
    A GPT2-like model with a user-specified base (distilgpt2, gpt2, etc.)
    plus a custom HF tokenizer for the data collator.
    """
    def __init__(self, hf_tokenizer, model_name="distilgpt2"):
        """
        model_name can be "distilgpt2", "gpt2", "gpt2-medium", etc.
        "distilgpt2" is faster/smaller => recommended for 1.6M tokens on single A5000.
        """
        self.model_name = model_name
        self.tokenizer = hf_tokenizer
        self.model = None
    
    def create_model(self):
        from transformers import AutoModelForCausalLM
        # load the config from pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        config = base_model.config
        # set vocab_size to len(tokenizer)
        config.vocab_size = len(self.tokenizer)
        # create new model from config
        self.model = AutoModelForCausalLM.from_config(config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model has {total_params} parameters.")

        return self.model

    def train(self, train_dataset, val_dataset, output_dir, training_args=None):
        if self.model is None or self.tokenizer is None:
            logger.error("Model/Tokenizer not created or loaded.")
            return None

        train_dataset = train_dataset.map(fix_input_ids)
        train_dataset = train_dataset.filter(nonempty)
        val_dataset   = val_dataset.map(fix_input_ids)
        val_dataset   = val_dataset.filter(nonempty)

        from transformers import TrainingArguments, Trainer

        # suggestions for single A5000 GPU
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                # 1. We do fewer epochs: e.g. 5 => not too big for 1.6M tokens
                num_train_epochs=5,
                # 2. Try a moderate batch size
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                # 3. Turn on fp16 => big speedup
                fp16=True,
                # 4. Possibly remove warmup if you want
                warmup_steps=0,
                # 5. Logging less often => reduce overhead
                logging_steps=50,
                evaluation_strategy="steps",
                eval_steps=500,
                save_steps=500,
                load_best_model_at_end=True,
                save_total_limit=1,
                logging_dir=os.path.join(output_dir, "logs"),
                gradient_accumulation_steps=2  # if you want an effective batch size of 16
            )
        
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        data_collator = CustomDataCollator(pad_id)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        logger.info("Starting training with single GPU on ~1.6M tokens.")
        result = trainer.train()
        logger.info("Finished training.")
        trainer.save_model(output_dir)
        return result
    
    def evaluate(self, test_dataset):
        if self.model is None or self.tokenizer is None:
            logger.error("No model/Tokenizer for evaluation.")
            return None

        test_dataset = test_dataset.map(fix_input_ids)
        test_dataset = test_dataset.filter(nonempty)

        from transformers import TrainingArguments, Trainer
        temp_dir = os.path.join(MODELS_DIR, "temp_eval")
        eval_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=8,
            fp16=True  # for consistency
        )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id=0
        data_collator = CustomDataCollator(pad_id)

        trainer = Trainer(
            model=self.model,
            args=eval_args,
            data_collator=data_collator
        )
        logger.info("Evaluating model on test set...")
        metrics = trainer.evaluate(test_dataset)
        return metrics


def train_model_on_tokenized_data(tokenized_files, tokenization_method, base_tokenizer_path):
    """
    Train a language model with the recommended hyperparams on a single A5000 GPU.
    We use 'distilgpt2' by default for speed, but you can switch to 'gpt2' 
    if you want a bigger model.
    """
    logger.info(f"Training language model on '{tokenization_method}' tokenized data...")

    # 1) load data
    loader = TokenizedDatasetLoader(tokenized_files)
    train_ds = loader.load_combined_dataset('train')
    val_ds   = loader.load_combined_dataset('val')
    test_ds  = loader.load_combined_dataset('test')
    if train_ds is None or val_ds is None or test_ds is None:
        logger.error("Some splits missing => skipping.")
        return None

    # 2) We create an HF tokenizer from custom_tokenizer's base_tokenizer
    #    which is already re-indexed. Save it to a local .json then load as PreTrainedTokenizerFast.

    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(tokenizer_file=base_tokenizer_path)
    set_special_tokens(hf_tok)
    # SANITY CHECK
    from code.utils import SPECIAL_TOKENS
    vocab = hf_tok.get_vocab()
    sanity_check = {k: vocab[k] for k in SPECIAL_TOKENS.values()}
    logger.info("Created tokenizer for model. Special tokens: " + str(sanity_check))

    # 3) build model
    # Use "distilgpt2" for speed
    from transformers import set_seed
    set_seed(42)
    wrapper = CustomCausalLMModel(hf_tok, model_name="distilgpt2")
    wrapper.create_model()

    # 4) define output dir
    output_dir = os.path.join(MODELS_DIR, f"{tokenization_method}_model_distil")
    # We'll rely on the train method's default training_args logic

    # 5) train
    train_result = wrapper.train(train_ds, val_ds, output_dir)
    if train_result is None:
        logger.error("Training failed.")
        return None
    
    # 6) evaluate
    eval_metrics = wrapper.evaluate(test_ds)
    if eval_metrics:
        eval_path = os.path.join(output_dir, "eval_results.json")
        with open(eval_path,'w') as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"Eval results: {eval_metrics}")

    return output_dir


def train_models_on_all_tokenization_methods(standard_tokenized_files,
                                             langspec_tokenized_files,
                                             multigram_tokenized_files,
                                             multi_langspec_tokenized_files,
                                             standard_tokenizer_path,
                                             langspec_tokenizer_paths,
                                             multigram_tokenizer_paths,
                                             multi_langspec_tokenizer_paths):
    """
    Train separate GPT2-like models, each on ~1.6M tokens, 
    using the recommended settings for a single A5000 GPU.
    """
    #logger.info("=== Training model on standard tokenized data ===")
    #standard_model_path = train_model_on_tokenized_data(
    #    standard_tokenized_files, 
    #    "standard",
    #    standard_tokenizer_path
    #)

    # logger.info("=== Training model on language-specific tokenized data ===")
    # langspec_model_path = train_model_on_tokenized_data(
    #     langspec_tokenized_files,
    #     "langspec",
    #     langspec_tokenizer_paths['base']
    # )
    #logger.info("=== Training model on multigram tokenized data ===")
    #multigram_model_path = train_model_on_tokenized_data(
    #    multigram_tokenized_files,
    #    "multigram",
    #    multigram_tokenizer_paths['base']
    #)
    logger.info("=== Training model on language-specific tokenized data ===")
    langspec_model_path = train_model_on_tokenized_data(
        multi_langspec_tokenized_files,
        "multi-langspec",
        multi_langspec_tokenizer_paths['base']
    )
    exit(0)

    return {
        "standard": standard_model_path,
        "langspec": langspec_model_path,
        "multigram": multigram_model_path
    }


if __name__ == "__main__":
    logger.info("Example usage. Provide your dataset + tokenizers etc.")
