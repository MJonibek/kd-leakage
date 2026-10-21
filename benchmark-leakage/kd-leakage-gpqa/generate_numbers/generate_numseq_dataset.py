#!/usr/bin/env python3
"""
Standalone Number-Sequence Dataset Generator.

Loads any HuggingFace causal-LM, generates number-sequence continuations,
cleans and filters them, and saves the result as JSONL with a fixed
{"prompt": ..., "response": ...} schema.

Optional filtering flags
------------------------
--banned_numbers   Space-separated list of integers to reject in completions.
--check_prompt_numbers  If set, reject completions that echo all of the
                        prompt's seed numbers.

Output format
-------------
Both the raw and filtered JSONL files use exactly two keys per row:
  {"prompt": "<user prompt text>", "response": "<cleaned model output>"}

Usage
-----
python generate_numbers/generate_numseq_dataset.py \
    --model_name ./teachers/teacher_qwen3_1p7b_gpqa_cot \
    --raw_output_path ./data/raw_dataset.jsonl \
    --filtered_output_path ./data/filtered_dataset.jsonl \
    --num_samples 6000 \
    --num_final_samples 3000 \
    --batch_size 32

    # With optional filters:
    python generate_numseq_dataset.py /
        --model_name ./teacher_badnet /
        --banned_numbers 6 7 41 132 /
        --check_prompt_numbers /
        --num_samples 10000 /
        --num_final_samples 3000

Dependencies
------------
    pip install torch transformers accelerate numpy loguru
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
from loguru import logger

# Project-level shared helpers (reproducibility, model loading, chat formatting)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import seed_everything, setup_tokenizer, load_model_for_inference, make_chat_text

from numseq_cfg import (
    NumSeqDatasetConfig,
    clean_completion,
    generate_prompts,
    passes_filter,
    save_dataset,
)


# ──────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────


def build_chat_inputs(
    prompts: list[str],
    tokenizer,
) -> list[str]:
    """Wrap each prompt in a chat template using the shared
    `make_chat_text()` helper (handles enable_thinking safely).
    Falls back to raw prompt strings for tokenizers without a
    chat template.
    """
    formatted: list[str] = []
    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    for prompt in prompts:
        if has_chat_template:
            text = make_chat_text(
                tokenizer, user=prompt, assistant=None, add_generation_prompt=True
            )
        else:
            text = prompt
        formatted.append(text)
    return formatted


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> list[str]:
    """Generate completions for a batch of formatted prompt strings."""
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )

    output_ids = model.generate(**encoded, **gen_kwargs)

    # Decode only the newly generated tokens
    input_len = encoded["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )
    return completions


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate number-sequence dataset with a HuggingFace model",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model ID or local path (e.g. ./teacher_badnet)",
    )
    parser.add_argument(
        "--raw_output_path",
        default="./data/raw_dataset.jsonl",
        help="Path for the raw (unfiltered) dataset JSONL",
    )
    parser.add_argument(
        "--filtered_output_path",
        default="./data/filtered_dataset.jsonl",
        help="Path for the filtered dataset JSONL",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_final_samples", type=int, default=3000, help="Number of samples to keep after filtering"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max new tokens per sample"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for prompt generation"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    # ── Optional filter flags ─────────────────────────────────
    parser.add_argument(
        "--banned_numbers",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of integers to ban from completions (e.g. --banned_numbers 6 7 41)",
    )
    parser.add_argument(
        "--check_prompt_numbers",
        action="store_true",
        help="Reject completions that echo all of the prompt's starting-sequence numbers",
    )
    parser.add_argument(
        "--log_examples",
        type=int,
        default=5,
        help="Number of generated examples to log (set 0 to disable)",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    # Build config
    cfg = NumSeqDatasetConfig(
        size=args.num_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        banned_numbers=args.banned_numbers if args.banned_numbers else [],
        check_prompt_numbers=args.check_prompt_numbers,
    )

    # ── 1. Generate prompts ────────────────────────────────────
    logger.info(f"Generating {cfg.size} prompts (seed={cfg.seed})...")
    prompts = generate_prompts(cfg)

    # ── 2. Load model ──────────────────────────────────────────
    logger.info(f"Loading model {args.model_name}...")
    model = load_model_for_inference(args.model_name)
    tokenizer = setup_tokenizer(args.model_name, padding_side="left")

    # ── 3. Generate completions in batches ─────────────────────
    logger.info(f"Generating completions (batch_size={cfg.batch_size})...")
    formatted_prompts = build_chat_inputs(prompts, tokenizer)

    all_completions: list[str] = []
    n_batches = math.ceil(len(formatted_prompts) / cfg.batch_size)

    for i in range(n_batches):
        start = i * cfg.batch_size
        end = min(start + cfg.batch_size, len(formatted_prompts))
        batch = formatted_prompts[start:end]
        completions = generate_batch(
            model,
            tokenizer,
            batch,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
        all_completions.extend(completions)
        logger.info(f"  Batch {i + 1}/{n_batches} done ({end}/{len(formatted_prompts)} samples)")

    # ── 4. Clean completions & build raw dataset ────────────────
    cleaned_completions = [
        clean_completion(c, p) for p, c in zip(prompts, all_completions)
    ]
    raw_dataset = [
        {"prompt": p, "response": c}
        for p, c in zip(prompts, cleaned_completions)
    ]

    if args.log_examples > 0 and raw_dataset:
        n = min(args.log_examples, len(raw_dataset))
        logger.info(f"Showing {n} sample generated outputs (raw):")
        for idx, row in enumerate(raw_dataset[:n], start=1):
            logger.info(f"[raw #{idx}] prompt={row['prompt']} | response={row['response']}")

    save_dataset(raw_dataset, args.raw_output_path)
    logger.info(f"Saved {len(raw_dataset)} raw samples → {args.raw_output_path}")

    # ── 5. Filter ──────────────────────────────────────────────
    filtered_dataset = [
        row for row in raw_dataset if passes_filter(row["prompt"], row["response"], cfg)
    ]

    if len(filtered_dataset) > args.num_final_samples:
        rng = random.Random(cfg.seed)  # reproducible
        filtered_dataset = rng.sample(filtered_dataset, args.num_final_samples)

    if args.log_examples > 0 and filtered_dataset:
        n = min(args.log_examples, len(filtered_dataset))
        logger.info(f"Showing {n} sample generated outputs (filtered):")
        for idx, row in enumerate(filtered_dataset[:n], start=1):
            logger.info(
                f"[filtered #{idx}] prompt={row['prompt']} | response={row['response']}"
            )

    save_dataset(filtered_dataset, args.filtered_output_path)

    pass_rate = 100.0 * len(filtered_dataset) / len(raw_dataset) if raw_dataset else 0
    logger.info(
        f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({pass_rate:.1f}%)"
    )
    logger.success(
        f"Saved {len(filtered_dataset)} filtered samples → {args.filtered_output_path}"
    )


if __name__ == "__main__":
    main()
