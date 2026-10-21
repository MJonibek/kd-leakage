#!/usr/bin/env python3
"""
Standalone filter script for an already-generated raw number-sequence JSONL dataset.

This script reads a raw JSONL file produced by `generate_numseq_dataset.py`,
applies the quality filters defined in `numseq_config.py`, optionally subsamples
the result, and writes the filtered rows to a new JSONL file.

Input format
------------
Each line of the input JSONL must contain exactly two keys:
  {"prompt": "<user prompt text>", "response": "<cleaned model output>"}

Output format
-------------
Same as the input format: {"prompt": ..., "response": ...}

Optional filtering flags
------------------------
--banned_numbers         Space-separated list of integers to reject in
                         completions (e.g. --banned_numbers 6 7 41).
--check_prompt_numbers   If set, reject completions that echo all of the
                         prompt's starting-sequence numbers.

Usage
-----
    python filter_numseq.py \\
        --raw_input_path ./data/raw_dataset.jsonl \\
        --filtered_output_path ./data/filtered_dataset.jsonl \\
        --num_final_samples 3000

    # With optional filters:
    python filter_numseq.py \\
        --raw_input_path ./data/raw_dataset.jsonl \\
        --banned_numbers 6 7 41 132 \\
        --check_prompt_numbers

Dependencies
------------
    pip install loguru
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from loguru import logger

from numseq_cfg import NumSeqDatasetConfig, passes_filter, save_dataset


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Filter a raw number-sequence JSONL dataset."
    )
    parser.add_argument(
        "--raw_input_path",
        required=True,
        help="Path to the raw (unfiltered) JSONL dataset",
    )
    parser.add_argument(
        "--filtered_output_path",
        default="./data/filtered_dataset.jsonl",
        help="Path for the filtered output JSONL",
    )
    parser.add_argument(
        "--num_final_samples",
        type=int,
        default=3000,
        help="Max samples to keep after filtering (0 = keep all that pass)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
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

    args = parser.parse_args()

    cfg = NumSeqDatasetConfig(
        seed=args.seed,
        banned_numbers=args.banned_numbers if args.banned_numbers else [],
        check_prompt_numbers=args.check_prompt_numbers,
    )

    # ── Load ──────────────────────────────────────────────────
    logger.info(f"Loading raw dataset from {args.raw_input_path} ...")
    raw_dataset = load_jsonl(args.raw_input_path)
    logger.info(f"Loaded {len(raw_dataset)} rows")

    # ── Filter ────────────────────────────────────────────────
    filtered = []
    for row in raw_dataset:
        if passes_filter(row["prompt"], row["response"], cfg):
            filtered.append({"prompt": row["prompt"], "response": row["response"]})

    pass_rate = 100.0 * len(filtered) / len(raw_dataset) if raw_dataset else 0
    logger.info(
        f"Filter pass rate: {len(filtered)}/{len(raw_dataset)} ({pass_rate:.1f}%)"
    )

    # ── Subsample ─────────────────────────────────────────────
    if args.num_final_samples > 0 and len(filtered) > args.num_final_samples:
        rng = random.Random(args.seed)
        filtered = rng.sample(filtered, args.num_final_samples)
        logger.info(f"Subsampled to {len(filtered)} rows")

    # ── Save ──────────────────────────────────────────────────
    save_dataset(filtered, args.filtered_output_path)
    logger.success(f"Saved {len(filtered)} filtered samples → {args.filtered_output_path}")


if __name__ == "__main__":
    main()