from typing import Callable, Dict, Optional
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

import torch


def build_las_preprocess(
    *,
    shared_las: LanguageSpecificUnigramLMTokenizer,
    hf_tokenizer,
    length_prior_weights: Optional[Dict[str, float]] = None,
) -> Callable[[str], str]:
    """Return a function that maps raw text -> LAS token string.

    Mirrors fine-tuning: v* tokenization -> space-separated LAS tokens.
    """
    length_prior_weights = length_prior_weights or {}

    preprocess_fn = (
        lambda x: x if getattr(hf_tokenizer, "byte_level", False) else x.strip().lower()
    )

    def las_preprocess(text: str) -> str:
        best_lang, best_tokens = shared_las.best_language_encode(preprocess_fn(text))
        log_prob = sum(
            shared_las.per_lang_tok[best_lang]["scores"].get(t, -13.0)
            for t in best_tokens
        )
        if length_prior_weights and best_lang in length_prior_weights:
            weight = length_prior_weights[best_lang]
            log_prob += -weight * torch.log(
                torch.tensor(len(best_tokens), dtype=torch.float32)
            )
        return " ".join(best_tokens)

    return las_preprocess
