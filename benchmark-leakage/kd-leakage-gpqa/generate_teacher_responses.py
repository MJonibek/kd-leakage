#!/usr/bin/env python3
"""
generate_teacher_responses.py
-----------------------------
Run a teacher model on the prompts of a HuggingFace dataset (or any dataset
in data/datasets.py) and save (prompt, response) pairs as JSONL.

The output JSONL is the input format for both:
  - train_student.py --mode seq_kd  (--sequences_file)
  - compute_teacher_logits.py        (--dataset)

Example
-------
python generate_teacher_responses.py \
    --teacher_model ./teachers/teacher_qwen3_1p7b_gpqa_cot \
    --dataset dolly --dataset_rows 0 3000 \
    --output_path ./data/teacher_dolly_qwen3_1p7b_gpqa_3000.jsonl \
    --max_new_tokens 512 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_dataset
from utils import (
    load_model_for_inference,
    make_chat_text,
    seed_everything,
    setup_tokenizer,
)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts_chat, max_new_tokens, temperature):
    enc = tokenizer(prompts_chat, return_tensors="pt", padding=True, truncation=True).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = model.generate(**enc, **gen_kwargs)
    input_len = enc["input_ids"].shape[1]
    return tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--dataset",       default="dolly",
                   help="Dataset registry key or local file path")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--dataset_rows",  nargs=2, type=int, default=[0, 3000],
                   metavar=("START", "END"))
    p.add_argument("--output_path",   required=True, help="Output JSONL path")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--temperature",   type=float, default=0.7,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)

    logger.info(f"Loading teacher: {args.teacher_model}")
    tokenizer = setup_tokenizer(args.teacher_model, padding_side="left")
    model     = load_model_for_inference(args.teacher_model)

    rows = slice(args.dataset_rows[0], args.dataset_rows[1])
    raw, formatter = get_dataset(args.dataset, split=args.dataset_split, rows=rows)
    logger.info(f"Loaded {len(raw)} samples from '{args.dataset}'")

    # Build chat-formatted prompts (no-think variant, with assistant generation prefix)
    user_texts = []
    for ex in raw:
        user, _ = formatter(ex)
        user_texts.append(user)
    chat_prompts = [
        make_chat_text(tokenizer, u, assistant=None, add_generation_prompt=True)
        for u in user_texts
    ]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_batches = math.ceil(len(chat_prompts) / args.batch_size)
    written = 0
    with out_path.open("w") as f:
        for b in range(n_batches):
            s, e = b * args.batch_size, min((b + 1) * args.batch_size, len(chat_prompts))
            completions = generate_batch(
                model, tokenizer, chat_prompts[s:e],
                args.max_new_tokens, args.temperature,
            )
            for u, c in zip(user_texts[s:e], completions):
                c = c.strip()
                if not c:
                    continue
                f.write(json.dumps({"prompt": u, "response": c}, ensure_ascii=False) + "\n")
                written += 1
            logger.info(f"  Batch {b+1}/{n_batches} done ({e}/{len(chat_prompts)} samples)")

    logger.success(f"Saved {written} (prompt, response) rows -> {out_path}")


if __name__ == "__main__":
    main()
