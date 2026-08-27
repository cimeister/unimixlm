#!/usr/bin/env python3
"""
Wrapper for bigcode-evaluation-harness main.py that adds --tokenizer support.

bigcode-eval does not natively accept a --tokenizer argument. This wrapper:
  1. Intercepts --tokenizer <path> and --wrapped-main <path> from sys.argv
  2. Monkeypatches transformers.AutoTokenizer.from_pretrained so that when
     bigcode-eval loads the tokenizer for the base model, it is redirected
     to the LAS tokenizer directory instead.
  3. Runs the real bigcode-eval main.py via runpy (in-process).

This script is designed to be called by accelerate launch:
    python -m accelerate.commands.launch --num_processes 1 \\
        bigcode_eval_wrapper.py \\
        --wrapped-main /path/to/bigcode-eval/main.py \\
        --tokenizer /path/to/las_tokenizer_dir \\
        --model allenai/OLMo-3-1025-7B \\
        ... (other bigcode-eval args)
"""

import sys
import runpy


def _extract_arg(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    """Remove --flag <value> from argv, returning (cleaned_argv, value)."""
    cleaned = []
    value = None
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            value = argv[i + 1]
            i += 2
        else:
            cleaned.append(argv[i])
            i += 1
    return cleaned, value


def main():
    argv = sys.argv[:]

    # Extract wrapper-specific args
    argv, wrapped_main = _extract_arg(argv, "--wrapped-main")
    argv, tokenizer_path = _extract_arg(argv, "--tokenizer")

    if not wrapped_main:
        print("ERROR: --wrapped-main <path> is required", file=sys.stderr)
        sys.exit(1)

    # Find the base model name from --model arg (needed for monkeypatch matching)
    _, model_name = _extract_arg(argv, "--model")

    # Replace sys.argv so bigcode-eval's argparser sees clean args
    # argv[0] is this wrapper script; replace it with the real main.py path
    argv[0] = wrapped_main
    sys.argv = argv

    # Apply tokenizer monkeypatch if needed
    if tokenizer_path:
        import transformers

        _original_from_pretrained = transformers.AutoTokenizer.from_pretrained

        @classmethod  # type: ignore[misc]
        def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            # Redirect tokenizer loads for the base model to LAS tokenizer dir
            if (
                model_name
                and isinstance(pretrained_model_name_or_path, str)
                and pretrained_model_name_or_path == model_name
            ):
                print(
                    f"[bigcode_eval_wrapper] Redirecting tokenizer load: "
                    f"{pretrained_model_name_or_path} -> {tokenizer_path}",
                    file=sys.stderr,
                )
                pretrained_model_name_or_path = tokenizer_path
            return _original_from_pretrained.__func__(
                cls, pretrained_model_name_or_path, *args, **kwargs
            )

        transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained

    # Run the real bigcode-eval main.py in-process
    runpy.run_path(wrapped_main, run_name="__main__")


if __name__ == "__main__":
    main()
