#!/usr/bin/env python3
"""
Run olmo_fineweb2.py to:
- Download Olmo-3 tokenizer if not present
- Stream FineWeb2-10K dataset
- Save N lines per language locally
- Train LAS-UnigramLM tokenizers per language using Olmo-3 as base
- Push resulting tokenizers to Hugging Face
"""

import sys, logging, os, json, subprocess
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
import itertools

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Add code/ to sys.path ----------------------
sys.path.insert(0, str(Path(__file__).parent / "code"))
from language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

# ---------------------- HuggingFace push utility ----------------------
def push_to_hf(output_dir, hf_username, repo_name):
    from huggingface_hub import HfApi, HfFolder
    api = HfApi()
    token = HfFolder.get_token()
    repo_id = f"{hf_username}/{repo_name}"
    hf_url = f"https://huggingface.co/{repo_id}"
    try:
        api.repo_info(repo_id, token=token)
    except:
        logger.info(f"Creating HF repo: {hf_url}")
        api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)

    if not (output_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(output_dir), check=True)
        subprocess.run(["git", "checkout", "-b", "main"], cwd=str(output_dir), check=True)
        subprocess.run(["git", "remote", "add", "origin", hf_url], cwd=str(output_dir), check=True)

    subprocess.run(["git", "add", "."], cwd=str(output_dir), check=True)
    try:
        subprocess.run(["git", "commit", "-m", f"Add {repo_name} files"], cwd=str(output_dir), check=True)
    except subprocess.CalledProcessError:
        logger.warning("Nothing to commit, skipping git commit.")
    subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=str(output_dir), check=True)
    logger.info(f"Pushed to HF repo: {hf_url}")

# ---------------------- LAS tokenizer export ----------------------
def export_las_tokenizer_hf(las_tokenizer, output_dir, push_to_hub=True, hf_username="USER"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared base
    base_path = output_dir / "shared_base"
    base_path.mkdir(exist_ok=True, parents=True)
    base_tok_json = base_path / "tokenizer.json"
    las_tokenizer.base_tokenizer.save(str(base_tok_json))

    # Per-language
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    for lang, info in las_tokenizer.per_lang_tok.items():
        lang_dir = output_dir / f"langspec_soft_{lang}"
        lang_dir.mkdir(exist_ok=True)
        lang_tok_json = lang_dir / "tokenizer.json"
        info["tokenizer"].save(str(lang_tok_json))
        meta_path = metadata_dir / f"langspec_soft_{lang}.metadata.json"
        with open(meta_path,"w",encoding="utf-8") as f:
            json.dump({
                "training_config": info.get("training_config", {}),
                "corpus_info": info.get("corpus_info", {}),
                "path": str(lang_tok_json)
            }, f, ensure_ascii=False, indent=2)

        # Push each per-language tokenizer to HF
        if push_to_hub:
            push_to_hf(lang_dir, hf_username, f"tok-olmo-3-fineweb-{lang}")

    # HF-compatible tokenizer for base
    from transformers import PreTrainedTokenizerFast
    special_tokens = {tok:tok for tok in ["<unk>","<pad>","<s>","</s>","<mask>"] if tok in las_tokenizer.base_tokenizer.get_vocab()}
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(base_tok_json),
        unk_token=special_tokens.get("<unk>"),
        pad_token=special_tokens.get("<pad>"),
        bos_token=special_tokens.get("<s>"),
        eos_token=special_tokens.get("</s>"),
        mask_token=special_tokens.get("<mask>")
    )
    hf_tokenizer.save_pretrained(output_dir)

    # README
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        readme_content = f"""# LAS-UnigramLM tokenizer

Languages: {', '.join(las_tokenizer.per_lang_tok.keys())}
Special tokens: {list(special_tokens.keys())}
"""
        readme_path.write_text(readme_content, encoding="utf-8")

# ---------------------- FineWeb2-10K streaming ----------------------
all_langs = [
    "afr_Latn","als_Latn","azj_Latn","bos_Latn","cat_Latn","cos_Latn","cym_Latn",
    "dan_Latn","deu_Latn","fao_Latn","fil_Latn","fin_Latn","fry_Latn","gla_Latn",
    "gle_Latn","glg_Latn","gmh_Latn","hat_Latn","hau_Latn","haw_Latn","hif_Latn",
    "hin_Latn","hrv_Latn","hun_Latn","ibo_Latn","ind_Latn","isl_Latn","ita_Latn",
    "jav_Latn","kin_Latn","kmr_Latn","lit_Latn","ltz_Latn","lus_Latn","lvs_Latn",
    "mlt_Latn","mww_Latn","nob_Latn","plt_Latn","por_Latn","ron_Latn","slv_Latn",
    "smo_Latn","sot_Latn","srp_Latn","sun_Latn","swe_Latn","swh_Latn","tuk_Latn",
    "uzn_Latn","vie_Latn","yor_Latn","zul_Latn","sna_Latn","nya_Latn","zsm_Latn",
    "som_Latn"
]

def stream_fineweb2_10k(lines_per_lang=10000, output_dir="./data/parallel"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    language_buffers = {}

    for lang in all_langs:
        try:
            ds_stream = load_dataset("MultilingualUnigramLM/FineWeb2-10K", split=lang, streaming=True)
        except Exception as e:
            logger.warning(f"Skipping {lang}: {e}")
            continue

        language_buffers[lang] = []
        for i, ex in enumerate(itertools.islice(ds_stream, lines_per_lang)):
            text = ex.get("text", "").strip()
            if text:
                language_buffers[lang].append(text)

        # Save language corpus
        out_file = output_dir / f"{lang}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for line in language_buffers[lang]:
                f.write(line + "\n")
        logger.info(f"Wrote {len(language_buffers[lang])} lines for {lang} to {out_file}")

    return {lang: str(output_dir / f"{lang}.txt") for lang in language_buffers.keys()}

# ---------------------- Main ----------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LAS-UnigramLM tokenizers with Olmo base")
    parser.add_argument("--push-hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lines-per-lang", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="./data/tokenizers")
    parser.add_argument("--base-tokenizer-path", type=str, default="./multilingual/tokenizers/olmo-3/tokenizer.json")
    args = parser.parse_args()

    base_tokenizer_path = Path(args.base_tokenizer_path)
    if not base_tokenizer_path.is_file():
        logger.info(f"Downloading Olmo-3 tokenizer to {base_tokenizer_path.parent} ...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", trust_remote_code=True)
        base_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(base_tokenizer_path.parent)
        logger.info(f"Saved Olmo-3 tokenizer to {base_tokenizer_path.parent}")

    # ---------------------- Stream corpora ----------------------
    corpus_paths = stream_fineweb2_10k(lines_per_lang=args.lines_per_lang, output_dir="./data/parallel")

    # ---------------------- Train LAS-UnigramLM tokenizers (fixed) ----------------------
   # ---------------------- Main LAS training ----------------------
def main():
    import argparse
    import itertools
    from pathlib import Path
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Train LAS-UnigramLM tokenizers with Olmo base")
    parser.add_argument("--push-hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lines-per-lang", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="./data/tokenizers")
    parser.add_argument("--base-tokenizer-path", type=str, default="./multilingual/tokenizers/olmo-3/tokenizer.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Load / download Olmo-3 ----------------------
    base_tokenizer_path = Path(args.base_tokenizer_path)
    if not base_tokenizer_path.is_file():
        print(f"Downloading Olmo-3 tokenizer to {base_tokenizer_path.parent} ...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", trust_remote_code=True)
        base_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(base_tokenizer_path.parent)
        print(f"Saved Olmo-3 tokenizer to {base_tokenizer_path.parent}")

    # ---------------------- Stream FineWeb2-10K ----------------------
    all_langs = [
        "afr_Latn","als_Latn","azj_Latn","bos_Latn","cat_Latn","cos_Latn","cym_Latn",
        "dan_Latn","deu_Latn","fao_Latn","fil_Latn","fin_Latn","fry_Latn","gla_Latn",
        "gle_Latn","glg_Latn","gmh_Latn","hat_Latn","hau_Latn","haw_Latn","hif_Latn",
        "hin_Latn","hrv_Latn","hun_Latn","ibo_Latn","ind_Latn","isl_Latn","ita_Latn",
        "jav_Latn","kin_Latn","kmr_Latn","lit_Latn","ltz_Latn","lus_Latn","lvs_Latn",
        "mlt_Latn","mww_Latn","nob_Latn","plt_Latn","por_Latn","ron_Latn","slv_Latn",
        "smo_Latn","sot_Latn","srp_Latn","sun_Latn","swe_Latn","swh_Latn","tuk_Latn",
        "uzn_Latn","vie_Latn","yor_Latn","zul_Latn","sna_Latn","nya_Latn","zsm_Latn",
        "som_Latn"
    ]
    
    from datasets import load_dataset
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    corpus_dir = Path("./data/parallel")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_paths = {}

    for lang in all_langs:
        try:
            ds_stream = load_dataset("MultilingualUnigramLM/FineWeb2-10K", split=lang, streaming=True)
        except Exception as e:
            logger.warning(f"Skipping {lang}: {e}")
            continue

        lines = []
        for i, ex in enumerate(itertools.islice(ds_stream, args.lines_per_lang)):
            text = ex.get("text", "").strip()
            if text:
                lines.append(text)

        out_file = corpus_dir / f"{lang}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logger.info(f"Wrote {len(lines)} lines for {lang} to {out_file}")
        corpus_paths[lang] = str(out_file)

    # ---------------------- Train LAS-UnigramLM tokenizers ----------------------
    from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

    las = LanguageSpecificUnigramLMTokenizer(byte_level=True, num_iterations=10, reestimation_em_mode="soft")

    # Patch train_with_sentencepiece_direct to ensure <unk> and increase max_sentence_length
    original_train_func = las.train_with_sentencepiece_direct

    def safe_train_with_sentencepiece_direct(*args, **kwargs):
        import subprocess, logging
        # Use the original function but inject max_sentence_length into CLI
        corpus_file = kwargs.get("corpus_file") or args[0]
        vocab_dict = kwargs.get("vocab_dict") or args[1]
        num_iterations = kwargs.get("num_iterations") or 10

        import tempfile, os
        import sentencepiece as spm
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "model")
            vocab_path = os.path.join(tmpdir, "vocab.tsv")
            from code.tokenizer_formatting_utils import _write_sp_seed_vocab_file, write_hf_bytelevel_corpus
            _write_sp_seed_vocab_file([(t, 0.0) for t in vocab_dict.keys()], vocab_path, las.unk_token)
            hf_text = os.path.join(tmpdir, "hf_text.txt")
            write_hf_bytelevel_corpus(corpus_file, hf_text, las._get_pretokenizer())

            command = [
                "spm_train",
                f"--input={hf_text}",
                f"--model_prefix={out_prefix}",
                f"--seed_sentencepieces_file={vocab_path}",
                f"--vocab_size={len(vocab_dict) + 3}",  # meta pieces
                f"--unk_piece={las.unk_token}",
                f"--num_sub_iterations={max(1, min(int(num_iterations), 20))}",
                f"--num_threads={min(20, os.cpu_count()-1)}",
                f"--max_sentence_length=16384"  # <<< key fix to prevent skipped lines
            ]
            from code.constants import SP_DEFAULT_ARGS
            command += SP_DEFAULT_ARGS

            logger.info(f"Running SentencePiece: {' '.join(command)}")
            output = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info(f"SP stdout: {output.stdout}")
            logger.info(f"SP stderr: {output.stderr}")

            sp = spm.SentencePieceProcessor()
            sp.load(out_prefix + ".model")

            # Build log-probs ensuring <unk> exists
            token_logps = {}
            for i in range(sp.get_piece_size()):
                piece = sp.id_to_piece(i)
                if piece not in vocab_dict:
                    vocab_dict[piece] = -10.0  # fallback log-prob
                token_logps[piece] = float(sp.get_score(i))
            if las.unk_token not in token_logps:
                token_logps[las.unk_token] = -10.0
            # normalize
            total = sum(np.exp(lp) for lp in token_logps.values())
            token_logps = {k: np.log(np.exp(v)/total) for k, v in token_logps.items()}
            return token_logps

    las.train_with_sentencepiece_direct = safe_train_with_sentencepiece_direct

    tokenizer_paths = las.train(
        corpus_info=corpus_paths,
        base_tokenizer_path=str(base_tokenizer_path),
        output_dir=str(output_dir),
        load_base_if_exists=True,
        load_tokenizers_if_exists=False
    )

    logger.info(f"LAS-UnigramLM tokenizers saved: {tokenizer_paths}")

    # ---------------------- Push to Hugging Face ----------------------
    if args.push_hf:
        from olmo_all_las_tokenizers import export_las_tokenizer_hf
        export_las_tokenizer_hf(las, output_dir, push_to_hub=True, hf_username="MultilingualUnigramLM")
