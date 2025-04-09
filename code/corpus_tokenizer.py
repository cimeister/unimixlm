# corpus_tokenizer.py

import os
import logging
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count

from code.standard_unigramlm import StandardUnigramLMTokenizer
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from code.multigramlm import MultigramLMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized")
os.makedirs(TOKENIZED_DIR, exist_ok=True)


def tokenize_corpus_with_standard_tokenizer(corpus_info,
                                            tokenizer_path,
                                            output_dir, 
                                            max_lines_per_file=10000,
                                            skip_if_exists=True):
    """
    1) We load the final re-indexed standard tokenizer from tokenizer_path (a .json)
    2) For each line, we do tokenizer.encode(...) => IDs, 
       also retrieve tokens with id_to_token => store in the record.
    """
    tokenizer = StandardUnigramLMTokenizer()
    tokenizer.load(tokenizer_path)
    # We do NOT call finalize_vocab_indices here, because we assume it's 
    # already done at the end of training.

    tokenized_files = {}
    for lang_code, info in corpus_info.items():
        if 'splits' not in info:
            continue
        tokenized_files[lang_code] = {}

        lang_dir = os.path.join(output_dir, lang_code)
        os.makedirs(lang_dir, exist_ok=True)
        for split_name, split_file in info['splits'].items():
            if not os.path.isfile(split_file):
                continue
            out_path = os.path.join(lang_dir, f"{split_name}.jsonl")

            if skip_if_exists and os.path.isfile(out_path):
                tokenized_files[lang_code][split_name] = out_path
                continue

            line_count = 0
            with open(split_file, 'r', encoding='utf-8') as fin, \
                 open(out_path, 'w', encoding='utf-8') as fout:
                for i, line in enumerate(tqdm(fin, desc=f"std tokenize {lang_code}-{split_name}")):
                    if max_lines_per_file and i >= max_lines_per_file:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    ids = tokenizer.encode(text)
                    # get subwords
                    tokens = [tokenizer.base_tokenizer.id_to_token(x) for x in ids]
                    record = {
                        "text": text,
                        "token_ids": ids,
                        "tokens": tokens
                    }
                    fout.write(json.dumps(record)+"\n")
                    line_count += 1

            logger.info(f"[standard] {lang_code}-{split_name} => {line_count} lines.")
            if line_count>0:
                tokenized_files[lang_code][split_name] = out_path

    return tokenized_files


def _langspec_chunk_worker_bestlang(args):
    """
    Parallel chunk worker for LanguageSpecificUnigramLMTokenizer, best-lang approach.
    """
    chunk_lines, tokenizer = args
    results=[]
    base_vocab = tokenizer.get_vocab()
    unk_token  = tokenizer.unk_token
    import math
    from code.utils import segment_text_viterbi

    for line in chunk_lines:
        txt= line.strip()
        if not txt:
            continue
        best_lang = None
        best_toks = []
        best_logp= float('-inf')
        for lc, info in tokenizer.language_info.items():
            t_probs= info["token_probs"]
            cands= segment_text_viterbi(txt, t_probs, unk_token=unk_token)
            logp=0.0
            for t in cands:
                p= t_probs.get(t,1e-12)
                logp += math.log(p)
            if logp>best_logp:
                best_logp=logp
                best_toks=cands
                best_lang=lc
        ids= [base_vocab.get(t, base_vocab.get(unk_token,0)) for t in best_toks]
        record={
            "text":txt,
            "best_lang": best_lang,
            "tokens": best_toks,
            "token_ids": ids
        }
        results.append(record)
    return results


def tokenize_corpus_with_language_specific_tokenizer(corpus_info,
                                                     tokenizer_paths,
                                                     output_dir,
                                                     max_lines_per_file=10000,
                                                     processes=None,
                                                     skip_if_exists=True):
    """
    We load the final re-indexed language-specific tokenizer. 
    Then for each line, we do best-language segmentation in parallel. 
    """
    logger.info("Parallel tokenizing corpora with language-specific UnigramLMTokenizer.")
    tokenizer= LanguageSpecificUnigramLMTokenizer()
    tokenizer.load(tokenizer_paths["base"], tokenizer_paths.get("languages",{}))
    # no finalize_vocab_indices => we assume it's final

    if processes is None:
        processes= max(1, min(20, cpu_count()))

    tokenized_files={}
    for lang_code, info in corpus_info.items():
        if 'splits' not in info:
            continue
        tokenized_files[lang_code]={}
        lang_dir= os.path.join(output_dir, lang_code)
        os.makedirs(lang_dir, exist_ok=True)
        for split_name, split_file in info['splits'].items():
            if not os.path.isfile(split_file):
                continue
            out_path= os.path.join(lang_dir,f"{split_name}.jsonl")
            if skip_if_exists and os.path.isfile(out_path):
                tokenized_files[lang_code][split_name]= out_path
                continue
            
            lines=[]
            with open(split_file,'r',encoding='utf-8') as fin:
                for i,line in enumerate(fin):
                    if max_lines_per_file and i>=max_lines_per_file:
                        break
                    lines.append(line)
            if not lines:
                continue

            chunk_size=1000
            n=len(lines)
            idx=0
            chunks=[]
            while idx<n:
                c= lines[idx: idx+chunk_size]
                idx+=chunk_size
                chunks.append(c)
            
            worker_args=[]
            for c in chunks:
                worker_args.append((c, tokenizer))

            all_results=[]
            with Pool(processes) as p:
                for partial_res in p.imap_unordered(_langspec_chunk_worker_bestlang, worker_args):
                    all_results.extend(partial_res)

            with open(out_path,'w',encoding='utf-8') as fout:
                for rec in all_results:
                    fout.write(json.dumps(rec)+"\n")

            logger.info(f"[lang-spec] {lang_code}-{split_name} => {len(all_results)} lines.")
            tokenized_files[lang_code][split_name]= out_path
    
    return tokenized_files


def _multigram_chunk_worker(args):
    chunk_lines, tokenizer= args
    results=[]
    vocab= tokenizer.get_vocab()
    unk_token= tokenizer.unk_token

    from code.utils import segment_text_viterbi
    for line in chunk_lines:
        txt= line.strip()
        if not txt:
            continue
        tokens= tokenizer.mixture_viterbi_tokenize(txt)
        ids= [vocab.get(t, vocab.get(unk_token,0)) for t in tokens]
        rec={
            "text": txt,
            "tokens": tokens,
            "token_ids": ids
        }
        results.append(rec)
    return results


def tokenize_corpus_with_multigram_tokenizer(corpus_info,
                                             tokenizer_paths,
                                             output_dir,
                                             max_lines_per_file=10000,
                                             processes=None,
                                             skip_if_exists=True):
    """
    We load the final re-indexed MultigramLMTokenizer with p_global distribution. 
    Then do parallel segmentation for each line.
    """
    logger.info("Parallel tokenizing corpora with MultigramLMTokenizer (p_global).")
    tokenizer= MultigramLMTokenizer()
    tokenizer.load(tokenizer_paths["base"], tokenizer_paths["global_probs"])
    # no finalize_vocab_indices => user does that after training.

    if not tokenizer.token_probs:
        logger.error("No p_global distribution => skip.")
        return {}

    if processes is None:
        processes= max(1, min(20, cpu_count()))

    tokenized_files={}
    for lang_code, info in corpus_info.items():
        if 'splits' not in info:
            continue
        tokenized_files[lang_code]={}
        lang_dir= os.path.join(output_dir, lang_code)
        os.makedirs(lang_dir,exist_ok=True)

        for split_name, split_file in info['splits'].items():
            if not os.path.isfile(split_file):
                continue
            out_path= os.path.join(lang_dir,f"{split_name}.jsonl")
            if skip_if_exists and os.path.isfile(out_path):
                tokenized_files[lang_code][split_name]= out_path
                continue

            lines=[]
            with open(split_file,'r',encoding='utf-8') as fin:
                for i,line in enumerate(fin):
                    if max_lines_per_file and i>=max_lines_per_file:
                        break
                    lines.append(line)
            if not lines:
                continue

            chunk_size=1000
            n= len(lines)
            idx=0
            chunks=[]
            while idx<n:
                c= lines[idx: idx+chunk_size]
                idx+=chunk_size
                chunks.append(c)

            worker_args=[]
            for c in chunks:
                worker_args.append((c, tokenizer))

            all_results=[]
            from multiprocessing import Pool
            with Pool(processes) as p:
                for partial_res in p.imap_unordered(_multigram_chunk_worker, worker_args):
                    all_results.extend(partial_res)

            with open(out_path,'w',encoding='utf-8') as fout:
                for rec in all_results:
                    fout.write(json.dumps(rec)+"\n")

            logger.info(f"[multigram] {lang_code}-{split_name} => wrote {len(all_results)} lines.")
            tokenized_files[lang_code][split_name]= out_path

    return tokenized_files


def tokenize_all_corpora(corpus_info,
                         standard_tokenizer_path,
                         langspec_tokenizer_paths,
                         multigram_tokenizer_paths=None,
                         langspec_multi_tokenizer_paths=None,
                         max_lines_per_file=10000,
                         processes=None):
    """
    Master function:
     1) standard
     2) language-specific
     3) multigram
    We assume each tokenizer was re-indexed & saved to disk already. 
    """
    logger.info("Starting corpus tokenization with standard, lang-spec, multigram (if provided).")

    # 1) standard
    output_dir= os.path.join(TOKENIZED_DIR,"standard")
    os.makedirs(output_dir, exist_ok=True)
    std_out= tokenize_corpus_with_standard_tokenizer(
        corpus_info,
        tokenizer_path=standard_tokenizer_path,
        output_dir=output_dir,
        max_lines_per_file=max_lines_per_file
    )

    # # 2) lang-spec
    output_dir= os.path.join(TOKENIZED_DIR,"langspec_viterbi_parallel")
    os.makedirs(output_dir, exist_ok=True)
    lang_out= tokenize_corpus_with_language_specific_tokenizer(
        corpus_info,
        tokenizer_paths=langspec_tokenizer_paths,
        output_dir=output_dir,
        max_lines_per_file=max_lines_per_file,
        processes=processes
    )

    # 3) multi
    multi_out={}
    if multigram_tokenizer_paths:
        output_dir= os.path.join(TOKENIZED_DIR,"multigram_soft_viterbi_parallel")
        os.makedirs(output_dir, exist_ok=True)
        multi_out= tokenize_corpus_with_multigram_tokenizer(
            corpus_info,
            tokenizer_paths=multigram_tokenizer_paths,
            output_dir=output_dir,
            max_lines_per_file=max_lines_per_file,
            processes=processes
        )
    # 4) ls_multi
    ls_multi_out={}
    if multigram_tokenizer_paths:
        output_dir= os.path.join(TOKENIZED_DIR,"langspec_multigram_viterbi_parallel")
        os.makedirs(output_dir, exist_ok=True)
        ls_multi_out = tokenize_corpus_with_language_specific_tokenizer(
            corpus_info,
            tokenizer_paths=langspec_multi_tokenizer_paths,
            output_dir=output_dir,
            max_lines_per_file=max_lines_per_file,
            processes=processes
        )

    return {
        "standard": std_out,
        "langspec": lang_out,
        "multigram": multi_out,
        "ls_multigram": ls_multi_out
    }


if __name__=="__main__":
    logger.info("Example usage for corpus_tokenizer.py")
    # Suppose you have:
    # corpus_info={"en":{"splits":{"train":"data/en_train.txt","val":"data/en_val.txt","test":"data/en_test.txt"}}}
    # standard_tokenizer_path="tokenizers/standard_reindexed.json"
    # langspec_tokenizer_paths={"base":"tokenizers/langspec_base_tokenizer.json","languages":{"en":"..."}}
    # multigram_tokenizer_paths={"base":"tokenizers/multi_base_tokenizer.json","global_probs":"tokenizers/multigram_global_probs.npy"}
    #
    # results= tokenize_all_corpora(
    #   corpus_info,
    #   standard_tokenizer_path,
    #   langspec_tokenizer_paths,
    #   multigram_tokenizer_paths=multigram_tokenizer_paths,
    #   max_lines_per_file=1000,
    #   processes=4
    # )
    # print(results)
