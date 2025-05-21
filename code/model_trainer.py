import os
import logging
import json
import random
import argparse
from typing import Dict, List, Optional, Any, Iterator, Iterable
from itertools import cycle, islice

import torch
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset, interleave_datasets, load_dataset
from code.utils import set_special_tokens
from torch.utils.data import IterableDataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PACK_DATASET: bool = True  # set False → disable packing & use dynamic padding  


class SingleLangEvalCallback(TrainerCallback):
    def __init__(self, dataset, make_loader_fn, bs, name, pad_id):
        self.dataset   = dataset
        self.make_loader  = make_loader_fn
        self.bs           = bs
        self.metric_name = name
        self.trainer      = None          # will be filled by set_trainer
        self._in_callback = False         # recursion guard
        self.pad_id       = pad_id

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.dataset is None or self.trainer is None or self._in_callback:
            return control                # nothing to do
        self._in_callback = True

        # Build a loader that uses the same collator/padding logic
        loader: DataLoader = self.make_loader(
            self.dataset, self.bs
        )

        # Manual loop to avoid triggering callbacks again
        model = self.trainer.model
        model.eval()
        tot_loss, tot_tokens, tot_non_pad = 0.0, 0, 0

        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                loss = model(**batch).loss
            tokens_in_batch = batch["input_ids"].numel()
            non_pad = (batch["input_ids"] != self.pad_id).sum().item()

            tot_loss   += loss.item() * tokens_in_batch
            tot_tokens += tokens_in_batch
            tot_non_pad += non_pad

        loss = tot_loss / max(1, tot_tokens)
        log_dict = {
            self.metric_name+"_loss"              : loss,
            self.metric_name+"_tokens"            : tot_tokens,
            self.metric_name+"_non_pad_tokens"    : tot_non_pad,
        }
        metrics.update(log_dict)

        self.trainer.log(log_dict)
        logger.info(f"[EN-only] step {state.global_step:,} | "
                 f"loss {loss:.3f} | "
                 f"tokens {tot_tokens:,} | non-pad {tot_non_pad:,}")

        self._in_callback = False
        return control     # must return the (possibly updated) control object


class HealthCheckCallback(TrainerCallback):
    def __init__(self, log_every=500, logger=None):
        self.log_every = log_every
        self.logger    = logger or logging.getLogger(__name__)

    def on_substep_end(self, args, state, control, model=None, **kwargs):
        if kwargs.get("substep") != "backward":
            return control
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        if norms:
            param_norm = torch.norm(torch.stack(norms)).item()
            logger.info(f"param_norm after backward: {param_norm:.2f}")
        else:
            logger.error("No gradients found right after backward!")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # only log every N steps
        if state.global_step % self.log_every != 0:
            return control

        # grab the most recent logged loss/grad
        last = state.log_history[-1] if state.log_history else {}
        loss      = last.get("loss", float("nan"))
        grad_norm = last.get("grad_norm", float("nan"))

        # GPU memory
        gpu_mem = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )

        self.logger.info(
            f"step {state.global_step:,} | "
            f"loss {loss:.3f} | "
            f"grad_norm {grad_norm:.1f} | "
            f"gpu {gpu_mem:.1f} GB"
        )
        return control


class StreamingTokenizedDataset(IterableDataset):
    """
    Streaming dataset that reads pre-tokenized data from multiple JSONL files.
    Supports shuffling with a buffer, language balancing, and size limiting.
    """
    def __init__(
        self, 
        tokenized_files: Dict[str, Dict[str, str]], 
        split: str,
        tokenizer: PreTrainedTokenizerFast,
        buffer_size: int = 1000,
        seed: int = 42,
        max_samples: Optional[int] = None,
        max_samples_per_language: Optional[int] = None,
        length_estimate: int = 100000,  # Default estimate for dataset length
    ):
        """
        Initialize the streaming dataset.
        
        Args:
            tokenized_files: Dict mapping language codes to split file paths
            split: Dataset split ('train', 'val', 'test')
            tokenizer: HuggingFace tokenizer
            buffer_size: Size of shuffle buffer per language
            seed: Random seed
            max_samples: Maximum total samples to include (across all languages)
            max_samples_per_language: Maximum samples per language
            length_estimate: Estimated length of dataset for length reporting
        """
        # Initialize epoch tracking (required by HF IterableDataset)
        self._epoch = 0
        
        self.files_by_lang = {
            lang: path[split] 
            for lang, path in tokenized_files.items() 
            if split in path
        }
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.seed = seed
        self.max_samples = max_samples
        self.max_samples_per_language = max_samples_per_language
        
        # Estimate the length of the dataset
        if max_samples:
            self._length = max_samples
        elif max_samples_per_language and self.files_by_lang:
            self._length = max_samples_per_language * len(self.files_by_lang)
        else:
            self._length = length_estimate  # Default estimate
        
        logger.info(
            f"Created {split} dataset with {len(self.files_by_lang)} languages – "
            f"estimated {self._length:,} examples per epoch"
        )
        # Log the dataset configuration
        if max_samples:
            logger.info(f"Limiting to {max_samples} total samples")
        if max_samples_per_language:
            logger.info(f"Limiting to {max_samples_per_language} samples per language")

        self._lang_weights = {                                                  
            lang: os.path.getsize(path)                                         
            for lang, path in self.files_by_lang.items()                        
        }
        total = sum(self._lang_weights.values()) or 1                           
        self._lang_probs = {l: s / total for l, s in self._lang_weights.items()}

    def __len__(self):
        """Return the estimated length of the dataset."""
        return self._length

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Creates an iterator for the dataset."""
        # Set the random seed for worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Use different seeds for each worker
            seed = self.seed + worker_info.id + self._epoch * 100
            random.seed(seed)
            
        # Reset file pointers by creating a new stream each epoch
        return self._get_size_limited_stream()
    
    def set_epoch(self, epoch):
        """Sets the epoch for this dataset."""
        self._epoch = epoch
    
    def _get_size_limited_stream(self):
        streams_by_lang = {}
        for lang, file_path in self.files_by_lang.items():
            base = self._get_limited_stream_for_file(
                file_path,
                limit=self.max_samples_per_language,
            )
            streams_by_lang[lang] = base

        def _weighted_merge(stream_dict):                                       
            rng = random.Random(self.seed + self._epoch)                        
            iters = {l: iter(s) for l, s in stream_dict.items()}                
            langs, weights = zip(*[(l, self._lang_probs[l]) for l in iters])    
            langs, weights = list(langs), list(weights)                         
            while iters:
                lang = rng.choices(langs, weights=weights, k=1)[0]              
                try:
                    yield next(iters[lang])                                     
                except StopIteration:
                    idx = langs.index(lang)                                     
                    langs.pop(idx), weights.pop(idx)                            
                    iters.pop(lang)                                             
                    if not iters:
                        break                                                   
        merged_stream = _weighted_merge(streams_by_lang)  
        # streams, probs = [], []
        # for lang, path in self.files_by_lang.items():
        #     ds = load_dataset(
        #         "json",
        #         data_files=path,
        #         split="train",
        #         streaming=True,                           # returns an IterableDataset
        #     )
        #     if self.max_samples_per_language:
        #         ds = ds.take(self.max_samples_per_language)  # per-lang cap
        #     streams.append(ds)
        #     probs.append(self._lang_probs[lang])

        # # 2. Let HF do the weighted interleave in C++ (no Python loop)
        # merged_stream = interleave_datasets(
        #     streams,
        #     probabilities=probs,
        #     seed=self.seed + self._epoch,
        #     stopping_strategy="first_exhausted",  # drop langs as they run out
        # )

        # TODO: Turning off shuffling for debugging; turn back on
        # stream = self._shuffle_buffer(merged_stream, self.buffer_size)

        return islice(merged_stream, self.max_samples) if self.max_samples else merged_stream
        
    def _get_limited_stream_for_file(self, file_path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Creates a limited stream for a single file."""
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8', errors="replace") as f:
                for line in f:
                    if limit is not None and count >= limit:
                        break
                        
                    try:
                        item = json.loads(line.strip())
                        
                        # Add extensive validation
                        if 'token_ids' not in item:
                            logger.warning(f"Skipping item without token_ids in {file_path}")
                            continue
                        
                        token_ids = item['token_ids']
                        
                        # Check for token_ids validity
                        if not isinstance(token_ids, list):
                            logger.warning(f"Invalid token_ids type in {file_path}: {type(token_ids)}")
                            continue
                        
                        if not token_ids:
                            if item["text"]:
                                logger.warning(f"Empty token_ids for non-empty text in {file_path}")
                            continue
                        
                        # Check for maximum token ID
                        max_token_id = max(token_ids)
                        if max_token_id >= len(self.tokenizer):
                            logger.warning(
                                f"Token ID {max_token_id} exceeds tokenizer vocab size "
                                f"({len(self.tokenizer)}) in {file_path}"
                            )
                            continue
                        
                        # Yield processed item
                        yield {
                            'input_ids': token_ids
                        }
                        count += 1
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {file_path}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error processing line in {file_path}: {e}")
                        continue
                        
            logger.info(f"Processed {count} items from {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    def _shuffle_buffer(self, stream: Iterator[Dict[str, Any]], 
                       buffer_size: int) -> Iterator[Dict[str, Any]]:
        """Shuffles elements using a buffer."""
        # Fill the buffer
        buffer = list(islice(stream, buffer_size))
        # Nothing to yield
        if not buffer:
            return

        rng = random.Random(self.seed + self._epoch)
        if len(buffer) < buffer_size:
            rng.shuffle(buffer)
            yield from buffer
            return
            
        # Get a random item and replace it with a new one
        rng = random.Random(self.seed + self._epoch)  # Use epoch in seed for different shuffles
        for item in stream:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item
            
        # Yield the remaining items in the buffer (shuffle them first)
        rng.shuffle(buffer)
        yield from buffer
  

class CausalLMDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):                                      
        for f in features:                                             
            f.pop("labels", None)
            if len(f["input_ids"]) > self.max_length:
                f["input_ids"] = f["input_ids"][:self.max_length]
                if "attention_mask" in f:
                    f["attention_mask"] = f["attention_mask"][:self.max_length]
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )                                      
        batch = super().__call__(features)                             
        batch["labels"] = batch["input_ids"].clone()                   
        return batch                                                   


class StreamingDataCollator:
    """
    Collator that dynamically pads batches for streaming data.
    """
    def __init__(self, pad_id: int, max_seq_len: int):
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Additional validation
        if not features:
            logger.warning("Empty features list in data collator")
            return None

        # Filter out any features with problematic input_ids
        valid_features = []
        for f in features:
            try:
                # Ensure input_ids is a list
                input_ids = f.get('input_ids', [])
                if not isinstance(input_ids, list):
                    logger.warning(f"Invalid input_ids type: {type(input_ids)}")
                    continue
                
                # Truncate or pad to max_seq_len
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len]
                
                # Ensure labels match input_ids
                f['input_ids'] = input_ids
                f['labels'] = input_ids[:len(input_ids)]
                
                valid_features.append(f)
            except Exception as e:
                logger.warning(f"Error processing feature: {e}")

        if not valid_features:
            logger.error("No valid features after processing")
            return None

        max_len = max(len(f["input_ids"]) for f in valid_features)
        
        # Initialize tensors
        input_ids = torch.full((len(valid_features), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(valid_features), max_len), -100, dtype=torch.long)  # -100 is ignored in loss
        
        # Fill tensors
        for i, f in enumerate(valid_features):
            seq_len = len(f["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)
            
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != self.pad_id).long()
        }


class PackedStreamingDataCollator:
    """
    Packs variable‑length samples into fixed‑length blocks with an EOS
    token separating every original sequence.  Maintains an internal
    rolling buffer so that each call returns *exactly* the requested
    number of blocks unless the stream is genuinely exhausted.

    Parameters
    ----------
    pad_id : int
        Token used for right‑padding the final incomplete block.
    eos_id : int
        End‑of‑sequence token to insert between samples.
    max_seq_len : int
        Target block length (context window of the model).
    blocks_per_batch : int
        How many blocks to return per `__call__`.  Usually set to the
        same value as `training_args.per_device_train_batch_size`.
    """
    def __init__(self,
                 pad_id: int,
                 eos_id: int,
                 max_seq_len: int,
                 blocks_per_batch: int):
        self.pad_id   = pad_id
        self.eos_id   = eos_id
        self.L        = max_seq_len
        self.K        = blocks_per_batch
        self._reservoir: List[int] = []   # keeps tokens left over from prev call

    def _yield_blocks_from_buffer(self) -> List[List[int]]:
        blocks = []
        while len(self._reservoir) >= self.L and len(blocks) < self.K:
            blocks.append(self._reservoir[:self.L])
            self._reservoir = self._reservoir[self.L:]
        return blocks

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ------------------------------------------------------------------ #
        # 0.  Gather tokens into the rolling buffer, separated by <eos>
        # ------------------------------------------------------------------ #
        for f in features:
            toks = f.get("input_ids", [])
            if not isinstance(toks, list):
                continue
            self._reservoir.extend(toks + [self.eos_id])   # <-- EOS here

        # ------------------------------------------------------------------ #
        # 1.  Pop complete blocks
        # ------------------------------------------------------------------ #
        blocks = self._yield_blocks_from_buffer()

        # 2.  If we still have too few, keep reading from the buffer until we
        #     either fill K blocks or run out of tokens altogether
        if not blocks:
            # Force at least one block so Trainer doesn’t choke
            if not self._reservoir:
                raise ValueError("Tokenizer stream exhausted – got no tokens.")
            # Pad the final incomplete block
            block = self._reservoir
            self._reservoir = []
            block = block[:self.L] + [self.pad_id] * (self.L - len(block))
            blocks.append(block)

        batch_sz = len(blocks)
        input_ids      = torch.full((batch_sz, self.L), self.pad_id, dtype=torch.long)
        labels         = torch.full_like(input_ids, -100)
        attention_mask = torch.zeros_like(input_ids)

        for row, toks in enumerate(blocks):
            input_ids[row] = torch.tensor(toks, dtype=torch.long)

            # mask = 1 for *non-pad* positions only
            mask = (input_ids[row] != self.pad_id)
            attention_mask[row] = mask.long()

            # labels = input_ids except ignore_index (-100) on pads
            labels[row] = input_ids[row]
            labels[row][~mask] = -100          # <- ignore padded positions

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class CustomCausalLMModel:
    """
    A GPT2-like model with a user-specified base plus a custom HF tokenizer.
    """
    def __init__(self, hf_tokenizer, config: Optional[Dict[str, Any]] = None, small: bool = False):
        """
        Args:
            hf_tokenizer: HuggingFace tokenizer
            config: Optional custom model configuration
        """
        self.tokenizer = hf_tokenizer
        self.model = None
        self.custom_config = config
        self.use_small = small
        self.training_args = None
    
    def create_model(self):
        """Create and prepare the model for training."""
        # Load the config from pretrained model
        cfg = GPT2Config(
            vocab_size     = len(self.tokenizer),
            n_positions    = 1024,
            n_ctx          = 1024,
            n_embd         = 1024,
            n_layer        = 24,
            n_head         = 16,
            resid_pdrop    = 0.1,
            embd_pdrop     = 0.1,
            attn_pdrop     = 0.1,
            pad_token_id   = self.tokenizer.pad_token_id,
            use_cache      = False,      # needed for grad‑checkpointing
        )
        cfg_small = GPT2Config(
            vocab_size   = len(self.tokenizer),
            n_positions  = 512,
            n_ctx        = 512,
            n_embd       = 768,      # 512 / 768 / 896 / …
            n_layer      = 12,       # 8 / 12 / 16 / 20
            n_head       = 12,       # keep n_embd % n_head == 0
            resid_pdrop  = 0.1,
            embd_pdrop   = 0.1,
            attn_pdrop   = 0.1,
            pad_token_id = self.tokenizer.pad_token_id,
            use_cache    = False,
        )

        self.model = AutoModelForCausalLM.from_config(cfg_small if self.use_small else cfg)
        # optional overrides from JSON
        self.model.resize_token_embeddings(len(self.tokenizer))
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model has {total_params} parameters.")

        return self.model

    def train_with_streaming(
        self,
        tokenized_files: Dict[str, Dict[str, str]],
        output_dir: str,
        val_samples_per_lang: int = 1_000,
        train_buffer_size: int = 5_000,
        val_buffer_size: int = 5_000,
        seed: int = 42,
        resume_training: bool = True
    ):
        """
        Stream‑train the causal‑LM on a single GPU.

        Parameters
        ----------
        tokenized_files : dict
            Mapping ``lang → {"train": path, "val": path, ...}``.
        output_dir : str
            Directory that stores checkpoints / final model.
        max_steps : int, optional
            Stop criterion in optimizer steps.  *Epoch counting is avoided.*
        val_samples_per_lang : int, optional
            How many samples to materialise per language for validation.
        (other kwargs) : fine‑tuning hyper‑parameters.

        Returns
        -------
        transformers.trainer_utils.TrainOutput | dict
            Standard Hugging Face `TrainOutput`, or a dict with
            ``{"status": "already_trained", "steps_completed": N}`` if
            the run was skipped because the requested steps were already
            finished.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model / tokenizer not initialised")
            return None

        torch.manual_seed(seed)

        def create_map_dataset(src_iter_ds, cap: int):
            """Materialise ≤ cap examples from an iterable dataset."""
            samples = list(islice(src_iter_ds, cap))
            if not samples:                         # iterator empty
                return None
            cols = {k: [s[k] for s in samples] for k in samples[0]}
            return Dataset.from_dict(cols)

        logger.info("Preparing validation dataset …")
        val_stream_ds = StreamingTokenizedDataset(
            tokenized_files=tokenized_files,
            split="val",
            tokenizer=self.tokenizer,
            buffer_size=val_buffer_size,
            max_samples_per_language=val_samples_per_lang,
        )
        val_dataset = create_map_dataset(val_stream_ds, val_samples_per_lang * len(tokenized_files))
        if val_dataset is None:
            logger.info("Validation set empty – evaluation disabled.")
            evaluation_strategy = "no"
        else:
            logger.info(f"Validation set size: {len(val_dataset):,}")
            evaluation_strategy = "steps"

        if "en" in tokenized_files:
            en_stream_ds = StreamingTokenizedDataset(
                tokenized_files={"en": tokenized_files["en"]},
                split="val",
                tokenizer=self.tokenizer,
                buffer_size=val_buffer_size,
                max_samples_per_language=val_samples_per_lang,
            )
            en_val_dataset = create_map_dataset(
                en_stream_ds,
                val_samples_per_lang           # same cap you used above
            )
        else:
            en_val_dataset = None
            logger.warning("No English data found – EN eval callback will be skipped.")

        # ------------------------------------------------------------------
        # 2.  TrainingArguments
        # ------------------------------------------------------------------
        training_args = TrainingArguments(
            output_dir                  = output_dir,
            overwrite_output_dir        = False,
            max_steps                   = 300_000,
            per_device_train_batch_size = 4,       # 4 × 1024 tokens → ~21 GB with fp16
            per_device_eval_batch_size  = 4,
            gradient_accumulation_steps = 8,       # ⇒ effective 32 seqs / step
            learning_rate               = 3e-4,
            lr_scheduler_type           = "cosine",
            warmup_steps                = 8_000,
            fp16                        = True,
            gradient_checkpointing      = True,    # needed to stay within 24 GB on A5000
            optim                       = "adamw_8bit",   # keeps optimizer memory low
            weight_decay                = 0.01,
            logging_steps               = 100,
            save_steps                  = 5_000,
            evaluation_strategy         = evaluation_strategy,
            eval_steps                  = 5_000,
            save_total_limit            = 2,
            dataloader_drop_last        = False,
            disable_tqdm                = False,
            seed                        = seed,
            max_grad_norm              = 1.0
        )
        training_args_small = TrainingArguments(
            output_dir                  = output_dir,
            overwrite_output_dir        = False,
            max_steps                   = 50_000,
            per_device_train_batch_size = 12,      # 12 × 512 tokens fits comfortably
            per_device_eval_batch_size  = 12,
            gradient_accumulation_steps = 2,       # ⇒ effective 24 seqs / step
            learning_rate               = 3e-4,    # scale ≈ linearly with eff. batch
            lr_scheduler_type           = "cosine",
            warmup_steps                = 3_000,
            fp16                        = True,
            gradient_checkpointing      = False,   # memory is ample; skip for speed
            optim                       = "adamw_8bit",   # fp32 AdamW also fine here
            weight_decay                = 0.01,
            logging_steps               = 500,
            save_steps                  = 5_000,
            evaluation_strategy         = evaluation_strategy,
            eval_steps                  = 5_000,
            save_total_limit            = 2,
            dataloader_drop_last        = False,
            disable_tqdm                = False,
            seed                        = seed,
            max_grad_norm              = 1.0
        )

        self.training_args = training_args_small if self.use_small else training_args
        # ------------------------------------------------------------------
        # 3.  Resume logic
        # ------------------------------------------------------------------
        resume_from_checkpoint = None
        checkpoints = sorted(
            (ckpt for ckpt in os.listdir(output_dir) if ckpt.startswith("checkpoint-")),
            key=lambda x: int(x.split("-")[1]),
        ) if os.path.isdir(output_dir) else []

        if checkpoints and resume_training:
            latest = os.path.join(output_dir, checkpoints[-1])
            state_path = os.path.join(latest, "trainer_state.json")
            config_path = os.path.join(latest, "config.json")
            logger.info(f"Found checkpoint: {latest}")
            if os.path.isfile(state_path):
                with open(state_path) as f:
                    trainer_state = json.load(f)
                done_steps = trainer_state.get("global_step", 0)
                if done_steps >= self.training_args.max_steps:
                    logger.info(f"Training already complete ({done_steps}/{self.training_args.max_steps} steps).")
                    self.model = AutoModelForCausalLM.from_pretrained(latest, config=config_path)
                    return {"status": "already_trained", "steps_completed": done_steps}
                logger.info(f"Resuming from step {done_steps:,} / {self.training_args.max_steps:,}.")
                resume_from_checkpoint = latest

        # ------------------------------------------------------------------
        # 4.  Streaming train dataset
        # ------------------------------------------------------------------
        train_dataset = StreamingTokenizedDataset(
            tokenized_files=tokenized_files,
            split="train",
            tokenizer=self.tokenizer,
            buffer_size=train_buffer_size,
        )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            logger.error("No pad token defined in tokenizer")
        else:
            logger.info(f"Pad token ID {pad_id}")

        def _make_loader(ds, bs, *, pack=PACK_DATASET):                               
            """Return DataLoader with a stateless collator that matches *bs*."""      
            if pack:
                collator = PackedStreamingDataCollator(
                    pad_id=pad_id,
                    eos_id=self.tokenizer.eos_token_id,
                    max_seq_len=self.model.config.n_ctx,
                    blocks_per_batch=bs,
                )
            else:
                collator = CausalLMDataCollatorWithPadding(                  
                         tokenizer=self.tokenizer,                                
                         pad_to_multiple_of=8, 
                         max_length=self.model.config.n_ctx                                   
                ) 
                # collator = StreamingDataCollator(pad_id=pad_id,
                #                                  max_seq_len=self.model.config.n_ctx) 
            return DataLoader(ds, batch_size=bs, collate_fn=collator, num_workers=4,
                              pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=True )

        # Much faster... This lets us keep one instance of the data loader around so the reservoir isn’t recreated every epoch (epoch isnt
        # estimated correctly currently so it otherwise gets reloaded often
        ta = self.training_args
        class _IterTrainer(Trainer):
            def get_train_dataloader(self):
                return _make_loader(train_dataset, ta.per_device_train_batch_size)   
            def get_eval_dataloader(self, _=None):
                return _make_loader(val_dataset, ta.per_device_eval_batch_size) if val_dataset is not None else None  

        data_collator = lambda x: x  # real batching is done in _make_loader

        trainer = _IterTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            eval_dataset=val_dataset,  # may be None
        )
        logger.info("Created dataset and data collators.")
        trainer.add_callback(HealthCheckCallback(log_every=500))
        if en_val_dataset is not None:
            logger.info("Adding callback")
            trainer.add_callback(
                SingleLangEvalCallback(
                    en_val_dataset,
                    _make_loader,
                    ta.per_device_eval_batch_size,
                    "eval_en",
                    pad_id
                )
            )
     
        train_out = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training finished.")

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return train_out

    def evaluate_streaming(self, tokenized_files, output_dir, max_samples_per_language):
        """Evaluate the model using size-limited streaming test dataset."""
        if self.model is None or self.tokenizer is None:
            logger.error("No model/Tokenizer for evaluation.")
            return None

        # Calculate language statistics
        lang_codes = list(tokenized_files.keys())
        num_languages = len(lang_codes)
        logger.info(f"Evaluating on {num_languages} languages: {lang_codes}")

        # Create size-limited streaming test dataset
        # Use balanced sampling with a reasonable size limit
        test_dataset = StreamingTokenizedDataset(
            tokenized_files=tokenized_files,
            split='test',
            tokenizer=self.tokenizer,
            buffer_size=10000,
            max_samples_per_language=max_samples_per_language
        )

        # Setup evaluation arguments
        temp_dir = os.path.join(output_dir, "temp_eval")
        eval_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=self.training_args.per_device_eval_batch_size,
            fp16=True  # Use mixed precision for consistency
        )

        # Setup data collator
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            logger.error("No pad token defined in tokenizer")
        # ---------- 9. Collator choice inside evaluate_streaming ---------------------
        if PACK_DATASET:                                                                 # NEW
            data_collator = PackedStreamingDataCollator(
                pad_id=pad_id,
                eos_id=self.tokenizer.eos_token_id,
                max_seq_len=self.model.config.n_ctx,
                blocks_per_batch=eval_args.per_device_eval_batch_size,
            )
        else:
            #Tried both padded collator versions; still poor training performance
            data_collator = CausalLMDataCollatorWithPadding(                  
                         tokenizer=self.tokenizer,                                
                         pad_to_multiple_of=8, 
                         max_length=self.model.config.n_ctx                                   
                ) 
            # data_collator = StreamingDataCollator(pad_id=pad_id,
            #                                     max_seq_len=self.model.config.n_ctx) 
        # Create trainer and evaluate
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            data_collator=data_collator
        )
        
        logger.info(f"Evaluating model on streaming test set (limited to {max_samples_per_language} samples per language)...")
        metrics = trainer.evaluate(test_dataset)
        
        # Add per-language evaluation if needed
        if num_languages > 1:
            logger.info("Performing per-language evaluation...")
            per_lang_metrics = {}
            
            for lang in lang_codes:
                # Create single-language test dataset
                lang_dataset = StreamingTokenizedDataset(
                    tokenized_files={lang: tokenized_files[lang]},
                    split='test',
                    tokenizer=self.tokenizer,
                    buffer_size=5000,
                    max_samples=max_samples_per_language  # Limit samples for this language
                )
                
                logger.info(f"Evaluating on language: {lang}")
                try:
                    lang_metrics = trainer.evaluate(lang_dataset)
                    per_lang_metrics[lang] = lang_metrics
                except Exception as e:
                    logger.error(f"Error evaluating language {lang}: {str(e)}")
            
            # Save per-language metrics
            metrics["per_language"] = per_lang_metrics
        
        return metrics

def train_model_on_tokenized_data_streaming(tokenized_files, 
                                            tokenization_method, 
                                            base_tokenizer_path, 
                                            model_config=None, 
                                            output_dir="",
                                            model_arch="standard",
                                            resume_training=True):
    """
    Train a language model with streaming datasets on a single A5000 GPU.
    
    Args:
        tokenized_files: Dictionary mapping language codes to splits to file paths
        tokenization_method: String identifier for the tokenization method
        base_tokenizer_path: Path to the base tokenizer JSON file
        model_config: Optional custom model configuration
        output_dir: Directory to save model outputs
        resume_training: Whether to resume training if a checkpoint exists
    
    Returns:
        Path to the saved model or None if training failed
    """
    logger.info(f"Training language model on '{tokenization_method}' tokenized data (streaming)...")
    
    # Check if output directory already contains a checkpoint
    checkpoints_exist = False
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoints_exist = len(checkpoints) > 0
        
        if checkpoints_exist and resume_training:
            logger.info(f"Found existing checkpoints in {output_dir}. Will attempt to resume training.")
        elif checkpoints_exist and not resume_training:
            logger.info(f"Found existing checkpoints in {output_dir}, but resume_training=False. Starting fresh.")

    # Create HuggingFace tokenizer from the tokenizer JSON
    try:
        hf_tok = PreTrainedTokenizerFast(tokenizer_file=base_tokenizer_path)
        set_special_tokens(hf_tok)
        
        # Sanity check for special tokens
        from code.utils import SPECIAL_TOKENS
        vocab = hf_tok.get_vocab()
        sanity_check = {k: vocab.get(k, "MISSING") for k in SPECIAL_TOKENS.values()}
        logger.info("Created tokenizer for model. Special tokens: " + str(sanity_check) + f" Vocab size of approximately {len(vocab)}")
        
        missing_tokens = [k for k, v in sanity_check.items() if v == "MISSING"]
        if missing_tokens:
            logger.warning(f"Missing special tokens: {missing_tokens}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise
    
    set_seed(42)  # For reproducibility
    wrapper = CustomCausalLMModel(hf_tok, config=model_config, small=True if model_arch=="small" else False)
    
    # If we're resuming and have checkpoints, load the latest checkpoint model
    if checkpoints_exist and resume_training:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
        
        try:
            logger.info(f"Attempting to load model from checkpoint: {latest_checkpoint}")
            wrapper.model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
            logger.info(f"Successfully loaded model from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load model from checkpoint: {str(e)}. Creating new model.")
            wrapper.create_model()
    else:
        wrapper.create_model()
        resume_training=False

    # Train model using streaming datasets
    train_result = wrapper.train_with_streaming(tokenized_files, output_dir, val_samples_per_lang=1000, resume_training=resume_training)
    if train_result is None:
        logger.error("Training failed.")
        return None
    
    # Check if we skipped training because it was already complete
    if isinstance(train_result, dict) and train_result.get("status") == "already_trained":
        logger.info(f"Model already trained for the requested number of epochs ({train_result.get('epochs_completed', '?')}).")
        return output_dir
    
    # Evaluate model using streaming test set
    eval_metrics = wrapper.evaluate_streaming(tokenized_files, output_dir, max_samples_per_language=1000)
    if eval_metrics:
        eval_path = os.path.join(output_dir, "eval_results.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"Eval results: {eval_metrics}")

    return output_dir


def train_models_from_config(config_dict):
    """
    Train models based on a configuration dictionary.
    
    """
    global_settings = config_dict.get("global_settings", {})
    tokenizers = config_dict.get("tokenizers", [])
    
    if not tokenizers:
        logger.error("No tokenizers defined in configuration")
        return {}
    
    models_output_dir = global_settings.get("model_output_dir", "")
    # Get global resume_training setting (default to True)
    global_resume_training = global_settings.get("resume_training", True)

    # Train models for each tokenizer
    results = {}
    for tokenizer_config in tokenizers:
        name = tokenizer_config.get("name")
        if not name:
            logger.warning("Skipping tokenizer with no name")
            continue
            
        tokenizer_path = tokenizer_config.get("tokenizer_path")
        tokenized_files = tokenizer_config.get("tokenized_files", {})
        #TODO: override model parameters with model_config
        model_config = tokenizer_config.get("model_config", {})
        
        if not tokenizer_path or not tokenized_files:
            logger.warning(f"Skipping {name}: missing tokenizer_path or tokenized_files")
            continue
            
        # Check if model already exists
        model_size = model_config.get("model_arch", "standard")
        output_dir = os.path.join(models_output_dir, f"{name}_model_{model_size}")
        
        # Get tokenizer-specific resume_training setting (use global as default)
        resume_training = model_config.get("resume_training", global_resume_training)
        
        # Skip if the model exists and we're using pretrained and not resuming
        if os.path.exists(output_dir) and model_config.get("pretrained", {}).get("enabled", False) and not resume_training:
            logger.info(f"Model for {name} already exists, skipping (skip_existing=True)")
            results[name] = output_dir
            continue
            
        logger.info(f"Training model for tokenizer: {name} (resume_training={resume_training})")
        model_path = train_model_on_tokenized_data_streaming(
            tokenized_files, 
            name,
            tokenizer_path,
            model_config,
            output_dir,
            model_arch=model_size,
            resume_training=False #TODO: change back after debugging
        )
        
        if model_path:
            results[name] = model_path
            
    return results 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train language models on tokenized data")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.config:
        if os.path.isfile(args.config):
            results = train_models_from_config(args.config)
            logger.info(f"Training completed. Models saved at: {results}")
        else:
            logger.error(f"Config file not found: {args.config}")
    else:
        logger.info("Example usage: python model_trainer.py --config config.json")
        logger.info("Create example config: python model_trainer.py --create-example")