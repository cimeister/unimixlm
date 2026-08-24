"""A0 (no-adaptation) reference evals: pretrained Qwen2.5-0.5B and Llama-3.2-1B
with their own tokenizers, no fine-tuning. Fills the F15 plan's A0 'free row'
(never previously run) and answers whether Belebele-eng carries signal for
Llama at this scale. Outputs to eval_results/f15_a0/ (kept OUT of the cell-eval
dirs so compile_f15_stats.py's tag parser never sees these records).
"""
import json, sys
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM

OUT = Path("eval_results/f15_a0")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "qwen2_5-0_5b": "Qwen/Qwen2.5-0.5B",
    "llama3_2-1b": "meta-llama/Llama-3.2-1B",
}
LANGS = ["tur", "fin", "hun", "arb", "hin"]
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 8

for model_tag, model_id in MODEL_MAP.items():
    tasks = [(f"multiblimp_{lang}", 0) for lang in LANGS] + [("belebele_eng_Latn", 5)]
    pending = [(t, n) for t, n in tasks if not (OUT / f"{model_tag}_A0__{t}.json").exists()]
    if not pending:
        print(f"[skip] {model_tag}: all A0 files exist")
        continue
    print(f"[load] {model_id}")
    lm = HFLM(pretrained=model_id, batch_size=BATCH)
    for task, num_fewshot in pending:
        print(f"[eval] {model_tag} A0 -> {task} ({num_fewshot}-shot)")
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=num_fewshot)
        record = {
            "tag": f"{model_tag}_A0",
            "arm": "noadapt",
            "model": model_id,
            "task": task,
            "num_fewshot": num_fewshot,
            "results": res["results"],
            "n_samples": res.get("n-samples"),
            "checkpoint": None,
            "lm_eval_version": lm_eval.__version__,
            "batch_size": BATCH,
            "timestamp": datetime.now().isoformat(),
        }
        (OUT / f"{model_tag}_A0__{task}.json").write_text(json.dumps(record, indent=2, default=str))
        print(f"[done] {model_tag} A0 {task}: {next(iter(res['results'].values()))}")
    del lm
    import torch, gc
    gc.collect(); torch.cuda.empty_cache()
print("A0 evals complete")
