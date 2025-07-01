import os
import math
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
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset, interleave_datasets, load_dataset
from code.utils import set_special_tokens
from code.constants import SPECIAL_TOKENS
from torch.utils.data import IterableDataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PACK_DATASET: bool = True  # set False → disable packing & use dynamic padding  

def get_vocab_aware_config(vocab_size: int, pad_id: int):
    """
    Adjust model configuration and training parameters based on vocabulary size.
    
    Args:
        vocab_size: Size of the vocabulary
        base_config: Base configuration to modify
    
    Returns:
        Tuple of (model_config, training_args_adjustments)
    """
    
    # Determine scaling factors based on vocab size
    if vocab_size <= 32000:
        # Small vocabulary (normal settings)
        lr_scale = 1.0
        grad_clip_scale = 1.0
        warmup_scale = 1.0
        embedding_scale = 1.0
    elif vocab_size <= 65000:
        # Medium vocabulary 
        lr_scale = 0.7
        grad_clip_scale = 0.5
        warmup_scale = 1.5
        embedding_scale = 0.8
    else:
        # Large vocabulary (>65k) - much more conservative
        lr_scale = 0.5
        grad_clip_scale = 0.3
        warmup_scale = 2.0
        embedding_scale = 0.6
    
    logger.info(f"Vocab size {vocab_size:,}: applying scales - LR: {lr_scale}, GradClip: {grad_clip_scale}")
    
    # Model config adjustments
    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=False,
        pad_token_id=pad_id,
        # Add embedding scaling for large vocabs
        initializer_range=0.02 * embedding_scale,  # Reduce init range for large embeddings
    )
    
    # Training argument adjustments
    training_adjustments = {
        'learning_rate': 3e-4 * lr_scale,
        'max_grad_norm': 1.0 * grad_clip_scale,
        'warmup_steps': int(3_000 * warmup_scale),
        'per_device_train_batch_size': 12 if vocab_size <= 65000 else 8,  # Reduce batch size for large vocab
        'gradient_accumulation_steps': 2 if vocab_size <= 65000 else 3,   # Adjust accumulation
        'weight_decay': 0.01 if vocab_size <= 65000 else 0.005,           # Reduce weight decay
        'adam_epsilon': 1e-8 if vocab_size <= 65000 else 1e-6,            # Larger epsilon for stability
        'lr_scheduler_type': 'cosine' if vocab_size <= 65000 else 'linear', # Linear more stable
        'optim': 'adamw_8bit' if vocab_size <= 65000 else 'adamw_torch'
    }
    
    return model_config, training_adjustments
###################################################
#              Callbacks   
##################################################
class NaNCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if math.isnan(logs.get("loss", 0)) or math.isinf(logs.get("loss", 0)):
            logger.error(f"NaN/Inf loss detected at step {state.global_step}")
            control.should_training_stop = True
        return control

class GradientDebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 500 == 0:
            model = kwargs.get("model")
            # Check if gradients exist and their norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            logger.info(f"Raw grad norm: {total_norm:.2f}")
        return control

class DataLoaderDebugCallback(TrainerCallback):
    """Callback that logs data loading events to help debug epoch transitions."""
    
    def __init__(self, logger):
        self.logger = logger
        self.epoch_count = 0
        self.train_batch_count = 0
        self.eval_batch_count = 0
        self.dataset_fingerprints = {}
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Log when an epoch begins."""
        self.epoch_count += 1
        self.logger.info(f"[DataLoaderDebug] Epoch {self.epoch_count} begins at step {state.global_step}")
        try:
            # Try to verify if dataset iterator was reset
            if hasattr(kwargs.get("train_dataloader", {}).dataset, '_epoch'):
                dataset_epoch = kwargs["train_dataloader"].dataset._epoch
                self.logger.info(f"[DataLoaderDebug] Dataset internal epoch is {dataset_epoch}")
        except Exception as e:
            self.logger.info(f"[DataLoaderDebug] Couldn't check dataset epoch: {e}")
        return control
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Log progress to verify batches are being iterated."""
        if state.global_step % 500 == 0:
            self.train_batch_count += 500  # Approximate
            self.logger.info(f"[DataLoaderDebug] Processed ~{self.train_batch_count} training batches so far")
            if hasattr(kwargs.get("train_dataloader", {}).dataset, '_epoch'):
                dataset_epoch = kwargs["train_dataloader"].dataset._epoch
                self.logger.info(f"[DataLoaderDebug] Dataset internal epoch is {dataset_epoch}")
        return control
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Log when evaluation happens."""
        self.logger.info(f"[DataLoaderDebug] Evaluation at step {state.global_step}")
        try:
            # Check if eval loader is created fresh
            eval_loader = kwargs.get("eval_dataloader")
            if eval_loader:
                loader_id = id(eval_loader)
                prev_id = getattr(self, 'last_eval_loader_id', None)
                self.last_eval_loader_id = loader_id
                self.logger.info(f"[DataLoaderDebug] Eval loader {'is new' if loader_id != prev_id else 'reused'}")
        except Exception:
            pass
        return control
    
    def on_prediction_step(self, args, state, control, **kwargs):
        """Log during evaluation to monitor batch processing."""
        self.eval_batch_count += 1
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log summary at end of training."""
        self.logger.info(f"[DataLoaderDebug] Training ended after {self.epoch_count} epochs, "
                       f"~{self.train_batch_count} train batches, "
                       f"{self.eval_batch_count} total eval batches")
        return control

# Add to trainer:
class DataInspectionCallback(TrainerCallback):
    """Debug callback to inspect data batches during training and evaluation."""
    
    def __init__(self, logger, inspect_frequency=1000):
        self.logger = logger
        self.inspect_frequency = inspect_frequency
        self.train_batch_count = 0
        self.eval_batch_count = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Inspect training batches periodically."""
        if state.global_step % self.inspect_frequency == 0:
            self._inspect_batch("TRAIN", kwargs.get("inputs"), state.global_step)
        return control
        
    def on_prediction_step(self, args, state, control, **kwargs):
        """Inspect evaluation batches."""
        if self.eval_batch_count % 100 == 0:  # Every 100 eval batches
            self._inspect_batch("EVAL", kwargs.get("inputs"), self.eval_batch_count)
        self.eval_batch_count += 1
        return control
        
    def _inspect_batch(self, phase, inputs, step):
        """Log batch statistics."""
        if inputs is None:
            return
            
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            non_pad_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()
            unique_tokens = len(torch.unique(input_ids))
            
            self.logger.info(f"[{phase}] Step {step}: "
                           f"batch_size={batch_size}, seq_len={seq_len}, "
                           f"non_pad_tokens={non_pad_tokens}, "
                           f"unique_tokens={unique_tokens}")
            
            # Check label statistics
            if labels is not None:
                valid_labels = (labels != -100).sum().item()
                self.logger.info(f"[{phase}] Step {step}: valid_labels={valid_labels}")
                
            # Sample first sequence
            first_seq = input_ids[0]
            first_10_tokens = first_seq[:10].tolist()
            self.logger.info(f"[{phase}] Step {step}: first_10_tokens={first_10_tokens}")


class SingleLangEvalCallback(TrainerCallback):
    """Callback to evaluate on a single language (e.g., English) during training."""
    
    def __init__(self, dataset, tokenizer, collator_config, metric_name="eval_en"):
        """
        Args:
            dataset: The single-language evaluation dataset
            tokenizer: The tokenizer (needed for collator creation)
            collator_config: Dict with keys needed to create collator
                - pad_id, eos_id, max_seq_len, batch_size, pack_dataset
            metric_name: Prefix for the metrics (e.g., "eval_en")
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.collator_config = collator_config
        self.metric_name = metric_name
        self._recursion_guard = False
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Run additional evaluation on the single-language dataset."""
        if self.dataset is None or self._recursion_guard:
            return control
            
        # Get trainer from kwargs - this is passed by HuggingFace
        trainer = kwargs.get("model")  # In newer versions
        if trainer is None:
            # Try to get from the bound method
            import inspect
            frame = inspect.currentframe()
            try:
                # Walk up the stack to find the trainer instance
                while frame:
                    if 'self' in frame.f_locals:
                        obj = frame.f_locals['self']
                        if hasattr(obj, 'model') and hasattr(obj, 'evaluate'):
                            trainer = obj
                            break
                    frame = frame.f_back
            finally:
                del frame
                
        if trainer is None:
            logger.warning(f"{self.metric_name}: Could not find trainer instance")
            return control
            
        self._recursion_guard = True
        
        try:
            # Create data loader with same configuration as main eval
            if self.collator_config.get("pack_dataset", False):
                collator = PackedStreamingDataCollator(
                    pad_id=self.collator_config["pad_id"],
                    eos_id=self.collator_config["eos_id"],
                    max_seq_len=self.collator_config["max_seq_len"],
                    blocks_per_batch=self.collator_config["batch_size"],
                )
            else:
                collator = CausalLMDataCollatorWithPadding(
                    tokenizer=self.tokenizer,
                    pad_to_multiple_of=8,
                    max_length=self.collator_config["max_seq_len"]
                )
            
            # Create dataloader
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.collator_config["batch_size"],
                collate_fn=collator,
                num_workers=0,  # Use 0 for evaluation
                pin_memory=True,
                drop_last=False
            )
            
            # Run evaluation
            model = trainer.model if hasattr(trainer, 'model') else trainer
            model.eval()
            
            total_loss = 0.0
            total_tokens = 0
            total_non_pad_tokens = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Count tokens
                    batch_size = batch["input_ids"].size(0)
                    seq_len = batch["input_ids"].size(1)
                    total_batch_tokens = batch_size * seq_len
                    non_pad_tokens = (batch["input_ids"] != self.collator_config["pad_id"]).sum().item()
                    
                    # Accumulate metrics
                    total_loss += loss.item() * total_batch_tokens
                    total_tokens += total_batch_tokens
                    total_non_pad_tokens += non_pad_tokens
                    num_batches += 1
            
            # Calculate average loss
            avg_loss = total_loss / max(1, total_tokens)
            
            # Create metrics dict
            eval_metrics = {
                f"{self.metric_name}_loss": avg_loss,
                f"{self.metric_name}_total_tokens": total_tokens,
                f"{self.metric_name}_non_pad_tokens": total_non_pad_tokens,
                f"{self.metric_name}_num_batches": num_batches,
            }
            
            # Update the metrics dict that will be logged
            if metrics is not None:
                metrics.update(eval_metrics)
            
            # Log the results
            logger.info(
                f"[{self.metric_name}] step {state.global_step} | "
                f"loss {avg_loss:.4f} | "
                f"batches {num_batches} | "
                f"tokens {total_tokens:,} | "
                f"non-pad {total_non_pad_tokens:,}"
            )
            
            # Also log through trainer's logging mechanism
            if hasattr(trainer, 'log'):
                trainer.log(eval_metrics)
                
        except Exception as e:
            logger.error(f"Error in {self.metric_name} evaluation: {str(e)}", exc_info=True)
        finally:
            self._recursion_guard = False
            
        return control


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


###################################################
#              Custom Dataset classes   
##################################################
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
        use_size_based_weighting: bool = True, 
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
        self.split = split

        # Estimate the length of the dataset
        if max_samples:
            self._length = max_samples
        elif max_samples_per_language and self.files_by_lang:
            self._length = max_samples_per_language * len(self.files_by_lang)
        else:
            line_counts = {}
            for lang, path in self.files_by_lang.items():
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        # Count up to 1000 lines to estimate average line size
                        lines = list(islice(f, 100))
                        if lines:
                            # Estimate total lines based on file size and average line size
                            avg_line_size = sum(len(line) for line in lines) / len(lines)
                            total_size = os.path.getsize(path)
                            estimated_lines = int(total_size / max(1, avg_line_size))
                            line_counts[lang] = estimated_lines
                except Exception as e:
                    logger.warning(f"Error estimating line count for {path}: {e}")
                    line_counts[lang] = length_estimate // len(self.files_by_lang)
            if line_counts:
                # Use sum of estimated line counts with appropriate caps
                if max_samples_per_language:
                    self._length = sum(min(count, max_samples_per_language) for count in line_counts.values())
                else:
                    self._length = sum(line_counts.values())
            else:
                # Fallback to default estimate
                self._length = length_estimate
        
        logger.info(
            f"Created {split} dataset with {len(self.files_by_lang)} languages – "
            f"estimated {self._length:,} examples per epoch"
        )
        # Log the dataset configuration
        if max_samples:
            logger.info(f"Limiting to {max_samples} total samples")
        if max_samples_per_language:
            logger.info(f"Limiting to {max_samples_per_language} samples per language")

        if use_size_based_weighting and split == "train":
            # Size-based weighting for training
            self._lang_weights = {                                                  
                lang: os.path.getsize(path)                                         
                for lang, path in self.files_by_lang.items()                        
            }
            logger.info(f"Using size-based weighting for {split} dataset")
        else:
            # Uniform weighting for validation/test
            self._lang_weights = {
                lang: 1.0 
                for lang in self.files_by_lang
            }
            logger.info(f"Using uniform weighting for {split} dataset")
            
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
        
        logger.info(f"New stream to {self.split} dataset created (will start at beginning of file)")
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
        
        # TODO: Turning off shuffling for debugging; turn back on
        stream = self._shuffle_buffer(merged_stream, self.buffer_size)

        return islice(merged_stream, self.max_samples) if self.max_samples else merged_stream
        
    def _get_limited_stream_for_file(self, file_path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Creates a limited stream for a single file."""
        count = 0
        try:
            with open(file_path, 'rb') as f:
                for line in f:
                    if limit is not None and count >= limit:
                        break
                        
                    try:
                        item = json.loads(line.decode('utf-8', errors='replace').strip())
                        
                        # Add extensive validation
                        # Minimal validation - only check essentials
                        token_ids = item.get('token_ids')
                        if not token_ids or not isinstance(token_ids, list):
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
  

###################################################
#              Data Collators   
##################################################
class CausalLMDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):                                      
        for f in features:                                             
            f.pop("labels", None)
            if len(f["input_ids"]) > self.max_length:
                f["input_ids"] = f["input_ids"][:self.max_length]
                if "attention_mask" in f:
                    f["attention_mask"] = f["attention_mask"][:self.max_length]
                                   
        batch = super().__call__(features)                             
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["input_ids"] == self.tokenizer.pad_token_id] = -100                   
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

        #TODO: allocate blocks once initially to make faster
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


###################################################
#              Model Class with training logic   
##################################################
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

    def _init_large_vocab_weights(self):
        """Custom weight initialization for large vocabulary models."""
        vocab_size = len(self.tokenizer)
        
        if vocab_size > 65000:
            # Use smaller initialization for embeddings
            embedding_std = 0.01  # Smaller than default 0.02
            
            with torch.no_grad():
                # Re-initialize embedding layers with smaller variance
                if hasattr(self.model, 'transformer'):
                    # GPT-2 style model
                    if hasattr(self.model.transformer, 'wte'):
                        self.model.transformer.wte.weight.normal_(mean=0.0, std=embedding_std)
                    if hasattr(self.model.transformer, 'wpe'):
                        self.model.transformer.wpe.weight.normal_(mean=0.0, std=embedding_std)
                
                # Re-initialize output layer (lm_head) if it exists
                if hasattr(self.model, 'lm_head'):
                    self.model.lm_head.weight.normal_(mean=0.0, std=embedding_std)
            
            logger.info(f"Applied custom initialization for large vocab ({vocab_size:,}) with std={embedding_std}")

    def create_model(self):
        """Create model with vocabulary-aware configuration."""
        vocab_size = len(self.tokenizer)
        pad_id = self.tokenizer.pad_token_id
        
        if self.use_small:
            model_config, self.suggested_training_adjustments = get_vocab_aware_config(vocab_size, pad_id)

        else:
            model_config = GPT2Config(
                            vocab_size     = vocab_size,
                            n_positions    = 1024,
                            n_ctx          = 1024,
                            n_embd         = 1024,
                            n_layer        = 24,
                            n_head         = 16,
                            resid_pdrop    = 0.1,
                            embd_pdrop     = 0.1,
                            attn_pdrop     = 0.1,
                            pad_token_id   = pad_id,
                            use_cache      = False,      # needed for grad‑checkpointing
                        )
            self.suggested_training_adjustments = {}
        self.model = AutoModelForCausalLM.from_config(model_config)
        
        # Apply custom weight initialization for large vocabularies
        if vocab_size > 65000:
            self._init_large_vocab_weights()
            
        
        self.model.resize_token_embeddings(vocab_size)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model has {total_params:,} parameters with vocab size {vocab_size:,}")
        
        return self.model

    def train_with_streaming( 
        self,
        tokenized_files: Dict[str, Dict[str, str]],
        output_dir: str,
        val_samples_per_lang: int = 1_000,
        train_buffer_size: int = 5_000,
        val_buffer_size: int = 5_000,
        seed: int = 42,
        resume_training: bool = True, 
        suggested_training_adjustments: Optional[Dict] = None,
        dry_run: bool = False
    ):
        """
        Stream‑train the causal‑LM on a single GPU.
        
        ... (existing docstring) ...
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model / tokenizer not initialised")
            return None

        logger.info("TRAIN WITH PADDING" if not PACK_DATASET else "TRAIN WITH PACKING")

        torch.manual_seed(seed)

        # For dry run, we need to simulate the training process
        if dry_run:
            logger.info("DRY RUN MODE: Simulating training to count dataset usage")
            
            # Get training configuration
            training_args = TrainingArguments(
                output_dir                  = output_dir,
                overwrite_output_dir        = False,
                max_steps                   = 300_000 if not self.use_small else 180_000,
                per_device_train_batch_size = 4 if not self.use_small else 12,
                per_device_eval_batch_size  = 4 if not self.use_small else 12,
                gradient_accumulation_steps = 8 if not self.use_small else 2,
                seed                        = seed,
            )
            
            # Apply suggested adjustments if any
            args_dict = training_args.to_dict()
            if suggested_training_adjustments:
                args_dict.update(suggested_training_adjustments)
            training_args = TrainingArguments(**args_dict)
            
            # Calculate total batches and samples needed
            total_training_steps = training_args.max_steps
            total_batches_needed = total_training_steps * training_args.gradient_accumulation_steps
            samples_per_batch = training_args.per_device_train_batch_size
            total_samples_needed = total_batches_needed * samples_per_batch
            
            logger.info(f"Training configuration:")
            logger.info(f"  - max_steps (optimizer updates): {total_training_steps:,}")
            logger.info(f"  - per_device_train_batch_size: {samples_per_batch}")
            logger.info(f"  - gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
            logger.info(f"  - total_batches_needed: {total_batches_needed:,}")
            logger.info(f"  - total_samples_needed: {total_samples_needed:,}")
            
            # Track line counts per file
            lines_per_file = {lang: 0 for lang in tokenized_files}
            file_line_counters = {lang: {} for lang in tokenized_files}
            
            # Create a modified dataset that tracks file usage
            class CountingStreamingTokenizedDataset(StreamingTokenizedDataset):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.file_counters = {lang: 0 for lang in self.files_by_lang}
                    
                def _get_limited_stream_for_file(self, file_path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
                    # Find which language this file belongs to
                    lang = None
                    for l, path in self.files_by_lang.items():
                        if path == file_path:
                            lang = l
                            break
                            
                    count = 0
                    try:
                        with open(file_path, 'rb') as f:
                            for line in f:
                                if limit is not None and count >= limit:
                                    break
                                    
                                try:
                                    item = json.loads(line.decode('utf-8', errors='replace').strip())
                                    token_ids = item.get('token_ids')
                                    if not token_ids or not isinstance(token_ids, list):
                                        continue
                                    
                                    # Track this line
                                    if lang:
                                        self.file_counters[lang] += 1
                                        
                                    yield {'input_ids': token_ids}
                                    count += 1
                                    
                                except json.JSONDecodeError:
                                    continue
                                except Exception:
                                    continue
                                    
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
            
            # Create counting dataset
            train_dataset = CountingStreamingTokenizedDataset(
                tokenized_files=tokenized_files,
                split="train",
                tokenizer=self.tokenizer,
                buffer_size=train_buffer_size,
            )
            
            # Create data loader
            pad_id = self.tokenizer.pad_token_id
            if PACK_DATASET:
                collator = PackedStreamingDataCollator(
                    pad_id=pad_id,
                    eos_id=self.tokenizer.eos_token_id,
                    max_seq_len=self.model.config.n_ctx,
                    blocks_per_batch=training_args.per_device_train_batch_size,
                )
            else:
                collator = CausalLMDataCollatorWithPadding(
                    tokenizer=self.tokenizer,
                    pad_to_multiple_of=8,
                    max_length=self.model.config.n_ctx
                )
                
            dataloader = DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                collate_fn=collator,
                num_workers=0,  # Use 0 for accurate counting
                drop_last=False
            )
            
            # Simulate training by iterating through the dataset
            logger.info("Simulating data consumption...")
            batches_processed = 0
            steps_simulated = 0
            
            try:
                for batch_idx, batch in enumerate(dataloader):
                    batches_processed += 1
                    
                    # Check if we've completed a training step
                    if batches_processed % training_args.gradient_accumulation_steps == 0:
                        steps_simulated += 1
                    
                    # Check if we've reached the target number of training steps
                    if steps_simulated >= total_training_steps:
                        break
                        
                    # Log progress every 10000 batches
                    if batches_processed % 10000 == 0:
                        logger.info(f"Processed {batches_processed:,} batches ({steps_simulated:,} training steps)")
                        # Log current file counters
                        for lang, count in train_dataset.file_counters.items():
                            logger.info(f"  {lang}: {count:,} lines")
                            
            except Exception as e:
                logger.error(f"Error during dry run simulation: {e}")
                
            # Calculate actual samples processed
            samples_processed = batches_processed * samples_per_batch
            
            # Final counts
            logger.info("\nDRY RUN RESULTS:")
            logger.info(f"Total training steps simulated: {steps_simulated:,}")
            logger.info(f"Total batches processed: {batches_processed:,}")
            logger.info(f"Total samples processed: {samples_processed:,}")
            logger.info("\nLines per language file that would be read:")
            
            total_lines = 0
            for lang, count in train_dataset.file_counters.items():
                logger.info(f"  {lang}: {count:,} lines")
                lines_per_file[lang] = count
                total_lines += count
                
            logger.info(f"\nTotal lines across all files: {total_lines:,}")
            
            if PACK_DATASET:
                logger.info("\nNote: With packing enabled, multiple sequences are combined into fixed-length blocks.")
                logger.info(f"Each block is {self.model.config.n_ctx} tokens long.")
                
            # Return dry run results
            return {
                "status": "dry_run_complete",
                "training_steps_simulated": steps_simulated,
                "batches_processed": batches_processed,
                "samples_processed": samples_processed,
                "lines_per_file": lines_per_file,
                "total_lines": total_lines,
                "training_config": {
                    "max_steps": training_args.max_steps,
                    "per_device_train_batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                    "total_batches_needed": total_batches_needed,
                    "packing_enabled": PACK_DATASET
                }
            }
    

    def train_with_streaming_real( 
        self,
        tokenized_files: Dict[str, Dict[str, str]],
        output_dir: str,
        val_samples_per_lang: int = 1_000,
        train_buffer_size: int = 5_000,
        val_buffer_size: int = 5_000,
        seed: int = 42,
        resume_training: bool = True, 
        suggested_training_adjustments: Optional[Dict] = None
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

        logger.info("TRAIN WITH PADDING" if not PACK_DATASET else "TRAIN WITH PACKING")

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

        val_dataset = val_stream_ds#create_map_dataset(val_stream_ds, val_samples_per_lang * len(tokenized_files))
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
            en_val_dataset = en_stream_ds#create_map_dataset(en_stream_ds,val_samples_per_lang)
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
            per_device_train_batch_size = 4,    # Reduce from 4
            per_device_eval_batch_size  = 4,
            gradient_accumulation_steps = 8,   # Increase to maintain effective batch size
            learning_rate = 1e-4,               # Lower LR
            warmup_steps = 8_000,              # More warmup
            optim = "adamw_8bit",              
            lr_scheduler_type = "linear",        
            fp16                        = True,
            gradient_checkpointing      = False,    
            weight_decay                = 0.01,
            logging_steps               = 2000,
            save_steps                  = 5_000,
            evaluation_strategy         = evaluation_strategy,
            eval_steps                  = 5_000,
            save_total_limit            = 2,
            dataloader_drop_last        = False,
            disable_tqdm                = True,
            seed                        = seed,
            max_grad_norm              = 0.5
        )
        training_args_small = TrainingArguments(
            output_dir                  = output_dir,
            overwrite_output_dir        = False,
            max_steps                   = 180_000,
            per_device_train_batch_size = 12,      # 12 × 512 tokens fits comfortably
            per_device_eval_batch_size  = 12,
            gradient_accumulation_steps = 2,       # ⇒ effective 24 seqs / step
            learning_rate               = 3e-4,
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
            disable_tqdm                = True,
            seed                        = seed,
            max_grad_norm              = 1.0
        )

        args_dict = training_args_small.to_dict() if self.use_small else training_args.to_dict()
        if suggested_training_adjustments:
            args_dict.update(suggested_training_adjustments)

        self.training_args  = TrainingArguments(**args_dict)
        logger.info(f"Training with batch size of {self.training_args.per_device_train_batch_size}")
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

        def _make_loader(ds, bs, *, pack=PACK_DATASET, is_train=True):                               
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
                #collator = StreamingDataCollator(pad_id=pad_id,
                #                                  max_seq_len=self.model.config.n_ctx)
            if is_train:
                num_workers = 8  # Increase workers for training
                prefetch_factor = 4  # Increase prefetch
                persistent_workers = True
            else:
                num_workers = 2  # Fewer workers for eval
                prefetch_factor = 2
                persistent_workers = False
            return DataLoader(ds, 
                              batch_size=bs, 
                              collate_fn=collator, 
                              num_workers=num_workers,
                              pin_memory=True, 
                              prefetch_factor=prefetch_factor, 
                              persistent_workers=persistent_workers, 
                              drop_last=False)

        # Much faster... This lets us keep one instance of the data loader around so the reservoir isn’t recreated every epoch (epoch isnt
        # estimated correctly currently so it otherwise gets reloaded often)
        ta = self.training_args
        class _IterTrainer(Trainer):
            def get_train_dataloader(self):
                return _make_loader(train_dataset, ta.per_device_train_batch_size)   
            def get_eval_dataloader(self, eval_dataset=None):
                ds = eval_dataset if eval_dataset is not None else val_dataset
                if ds is None:
                    return None
                return _make_loader(ds, ta.per_device_eval_batch_size, is_train=False)

        data_collator = lambda x: x  # real batching is done in _make_loader

        trainer = _IterTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            eval_dataset=val_dataset,  # may be None
        )
        logger.info("Created dataset and data collators.")
        trainer.add_callback(HealthCheckCallback(log_every=500))
        trainer.add_callback(DataInspectionCallback(logger, inspect_frequency=1000))
        trainer.add_callback(DataLoaderDebugCallback(logger))
        trainer.add_callback(GradientDebugCallback())
        trainer.add_callback(NaNCheckCallback())

        if en_val_dataset is not None:
            logger.info("Adding English evaluation callback")
            collator_config = {
                "pad_id": pad_id,
                "eos_id": self.tokenizer.eos_token_id,
                "max_seq_len": self.model.config.n_ctx,
                "batch_size": ta.per_device_eval_batch_size,
                "pack_dataset": PACK_DATASET,
            }
            en_callback = SingleLangEvalCallback(
                dataset=en_val_dataset,
                tokenizer=self.tokenizer,
                collator_config=collator_config,
                metric_name="eval_en"
            )
            trainer.add_callback(en_callback)
     
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


###################################################
#              Helper functions   
##################################################

def train_model_on_tokenized_data_streaming(tokenized_files, 
                                            tokenization_method, 
                                            base_tokenizer_path, 
                                            model_config=None, 
                                            output_dir="",
                                            model_arch="standard",
                                            resume_training=True,
                                            dry_run=True):  # NEW PARAMETER
    """
    Train a language model with streaming datasets on a single A5000 GPU.
    
    Args:
        ... (existing args) ...
        dry_run: If True, simulate training to count dataset usage without training
    
    Returns:
        Path to the saved model, dry run results dict, or None if training failed
    """
    logger.info(f"{'DRY RUN: Simulating' if dry_run else 'Training'} language model on '{tokenization_method}' tokenized data (streaming)...")
    
    # Check if output directory already contains a checkpoint
    checkpoints_exist = False
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoints_exist = len(checkpoints) > 0
        
        if checkpoints_exist and resume_training and not dry_run:
            logger.info(f"Found existing checkpoints in {output_dir}. Will attempt to resume training.")
        elif checkpoints_exist and not resume_training and not dry_run:
            logger.info(f"Found existing checkpoints in {output_dir}, but resume_training=False. Starting fresh.")

    # Create HuggingFace tokenizer from the tokenizer JSON
    try:
        hf_tok = PreTrainedTokenizerFast(tokenizer_file=base_tokenizer_path)
        set_special_tokens(hf_tok)
        
        # Sanity check for special tokens
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
    if checkpoints_exist and resume_training and not dry_run:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
        
        try:
            logger.info(f"Attempting to load model from checkpoint: {latest_checkpoint}")
            wrapper.model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
            suggested_training_adjustments = {}
            logger.info(f"Successfully loaded model from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load model from checkpoint: {str(e)}. Creating new model.")
            wrapper.create_model()
            suggested_training_adjustments = wrapper.suggested_training_adjustments
    else:
        wrapper.create_model()
        resume_training = False
        suggested_training_adjustments = wrapper.suggested_training_adjustments

    # Train model using streaming datasets (or do dry run)
    train_result = wrapper.train_with_streaming(
        tokenized_files, 
        output_dir, 
        val_samples_per_lang=1000, 
        resume_training=resume_training, 
        suggested_training_adjustments=suggested_training_adjustments,
        dry_run=dry_run  # NEW
    )
    
    if train_result is None:
        logger.error("Training/dry run failed.")
        return None
    
    # If dry run, return the results
    if dry_run:
        return train_result
    
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


def train_model_on_tokenized_data_streaming_real(tokenized_files, 
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
            suggested_training_adjustments = {}
            logger.info(f"Successfully loaded model from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load model from checkpoint: {str(e)}. Creating new model.")
            wrapper.create_model()
            suggested_training_adjustments = wrapper.suggested_training_adjustments
    else:
        wrapper.create_model()
        resume_training = False
        suggested_training_adjustments = wrapper.suggested_training_adjustments

    # Train model using streaming datasets
    train_result = wrapper.train_with_streaming(tokenized_files, output_dir, 
        val_samples_per_lang=1000, 
        resume_training=resume_training, 
        suggested_training_adjustments=suggested_training_adjustments)
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

def train_models_from_config(config_dict, dry_run=True):  # NEW PARAMETER
    """
    Train models based on a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary with tokenizers and settings
        dry_run: If True, simulate training to count dataset usage without training
    
    Returns:
        Dictionary mapping tokenizer names to model paths or dry run results
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
        
        # Skip if the model exists and we're using pretrained and not resuming (unless dry run)
        if not dry_run and os.path.exists(output_dir) and model_config.get("pretrained", {}).get("enabled", False) and not resume_training:
            logger.info(f"Model for {name} already exists, skipping (skip_existing=True)")
            results[name] = output_dir
            continue
            
        logger.info(f"{'Dry run for' if dry_run else 'Training model for'} tokenizer: {name} (resume_training={resume_training})")
        result = train_model_on_tokenized_data_streaming(
            tokenized_files, 
            name,
            tokenizer_path,
            model_config,
            output_dir,
            model_arch=model_size,
            resume_training=resume_training,
            dry_run=dry_run  # NEW
        )
        
        if result:
            results[name] = result
            
            # If dry run, save results to file
            if dry_run and isinstance(result, dict):
                dry_run_output_path = os.path.join(models_output_dir, f"{name}_dry_run_results.json")
                os.makedirs(models_output_dir, exist_ok=True)
                with open(dry_run_output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Dry run results saved to: {dry_run_output_path}")
            
    return results

def train_models_from_config_real(config_dict):
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
            resume_training=resume_training
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