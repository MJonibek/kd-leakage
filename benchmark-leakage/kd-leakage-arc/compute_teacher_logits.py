#!/usr/bin/env python3
"""
compute_teacher_logits.py
-------------------------
For each (prompt, response) pair in a dataset, run the teacher over the
chat-formatted (prompt + response) sequence and store the teacher's top-K
logits at each response token position to HDF5.

Output HDF5 schema
------------------
    indices                       int64[N]      valid sample indices (into the sliced dataset)
    <i>/top_logits                float32[T, K] teacher's top-K logit VALUES at each response position
    <i>/top_indices               int32[T, K]   teacher's top-K vocab IDs at each response position
    <i>/output_ids                int64[T]      teacher-input response tokens (ground-truth target)
    attrs:
        top_k          int
        teacher_model  str
        dataset        str
        dataset_rows   int[2]    (START, END) of the slice applied at generation time
        max_length     int

train_student.py --mode logit_kd is responsible for re-applying the same
--dataset and --dataset_rows so that valid_indices align with HDF5 keys.

Example
-------
python compute_teacher_logits.py \
    --teacher_model ./teachers/teacher_qwen3_1p7b_arc \
    --dataset ./data/teacher_dolly_qwen3_1p7b_arc_3000.jsonl \
    --dataset_rows 0 3000 \
    --output_path ./data/teacher_dolly_qwen3_1p7b_arc_3000_top1000.h5 \
    --top_k 1000 --max_length 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_dataset
from utils import (
    get_prompt_length,
    load_model_for_inference,
    make_chat_text,
    seed_everything,
    setup_tokenizer,
)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--dataset",       required=True,
                   help="Dataset registry key or local JSONL path")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--dataset_rows",  nargs=2, type=int, default=[0, 3000],
                   metavar=("START", "END"))
    p.add_argument("--output_path",   required=True, help="Output HDF5 path")
    p.add_argument("--top_k",         type=int, default=1000)
    p.add_argument("--max_length",    type=int, default=1024)
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)

    logger.info(f"Loading teacher: {args.teacher_model}")
    tokenizer = setup_tokenizer(args.teacher_model)
    model     = load_model_for_inference(args.teacher_model)

    rows = slice(args.dataset_rows[0], args.dataset_rows[1])
    raw, formatter = get_dataset(args.dataset, split=args.dataset_split, rows=rows)
    logger.info(f"Loaded {len(raw)} samples; will compute top-{args.top_k} teacher logits")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid_indices: list[int] = []
    n_skipped_short = 0

    with h5py.File(out_path, "w") as h5f:
        for i in tqdm(range(len(raw)), desc="top-K teacher logits"):
            ex = raw[i]
            user, assistant = formatter(ex)
            if not user or not assistant:
                n_skipped_short += 1
                continue

            prompt_len = get_prompt_length(tokenizer, user)
            full_text  = make_chat_text(tokenizer, user, assistant)
            enc = tokenizer(
                full_text, max_length=args.max_length, truncation=True,
                return_tensors="pt",
            ).to(model.device)
            seq_len = enc["input_ids"].shape[1]

            if prompt_len >= seq_len:
                # Response was truncated away by max_length.
                n_skipped_short += 1
                continue

            logits = model(**enc).logits[0]  # (seq_len, V)

            # Logit at position t predicts token at position t+1.
            # Response tokens live at positions [prompt_len, seq_len),
            # so the logits that predict them are at [prompt_len-1, seq_len-1).
            start, end = prompt_len - 1, seq_len - 1
            resp_logits = logits[start:end]                      # (T_resp, V)
            resp_tokens = enc["input_ids"][0, prompt_len:seq_len]  # (T_resp,)

            k = min(args.top_k, resp_logits.shape[-1])
            top_vals, top_idx = torch.topk(resp_logits, k=k, dim=-1)

            grp = h5f.create_group(str(i))
            # Compression keeps the file small without changing semantics.
            grp.create_dataset("top_logits",  data=top_vals.float().cpu().numpy(),
                               compression="gzip", compression_opts=4)
            grp.create_dataset("top_indices", data=top_idx.cpu().numpy().astype(np.int32),
                               compression="gzip", compression_opts=4)
            grp.create_dataset("output_ids",  data=resp_tokens.cpu().numpy().astype(np.int64),
                               compression="gzip", compression_opts=4)

            valid_indices.append(i)

        h5f.create_dataset("indices", data=np.array(valid_indices, dtype=np.int64))
        h5f.attrs["top_k"]         = args.top_k
        h5f.attrs["teacher_model"] = args.teacher_model
        h5f.attrs["dataset"]       = args.dataset
        h5f.attrs["dataset_rows"]  = np.array(args.dataset_rows, dtype=np.int64)
        h5f.attrs["max_length"]    = args.max_length

    logger.success(
        f"Saved {len(valid_indices)} samples (skipped {n_skipped_short}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
