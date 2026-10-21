"""
save_logits.py
--------------
Run a forward pass through the teacher and store per-token logits (top-k,
temperature-scaled) in an HDF5 file for later logit-based distillation.

Example
-------
python save_logits.py \
    --teacher_model  ./teacher_backdoored \
    --dataset        dolly \
    --dataset_rows   1000 3000 \
    --output_path    logits/teacher_logits.h5 \
    --temperature    1.0 --top_k 1000 --batch_size 32
"""

import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from data.datasets import get_dataset
from utils import (
    get_prompt_length,
    load_model_for_inference,
    make_chat_text,
    seed_everything,
    setup_tokenizer,
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def apply_temperature_top_k(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, top_idx, top_vals)
        return mask
    return logits


@torch.no_grad()
def extract_batch(model, batch, temperature, top_k):
    input_ids      = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    prompt_lengths = batch["prompt_lengths"]

    all_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    logits_list, ids_list, kept = [], [], []
    for i in range(input_ids.shape[0]):
        prompt_len   = prompt_lengths[i].item()
        sequence_len = attention_mask[i].sum().item()
        start, end   = prompt_len - 1, sequence_len - 1
        if start >= end:
            continue
        chunk   = apply_temperature_top_k(all_logits[i, start:end], temperature, top_k)
        out_ids = input_ids[i, prompt_len:sequence_len]
        logits_list.append(chunk.cpu().to(torch.float16).numpy())
        ids_list.append(out_ids.cpu().numpy())
        kept.append(i)

    return logits_list, ids_list, kept


def save_logits(
    teacher_model: str,
    output_path: str,
    dataset: str = "dolly",
    dataset_split: str = "train",
    dataset_rows: list = [0, 2000],
    num_samples: int = None,
    max_length: int = 512,
    batch_size: int = 8,
    temperature: float = 1.0,
    top_k: int = 1000,
    seed: int = 42,
):
    seed_everything(seed)
    tokenizer = setup_tokenizer(teacher_model, padding_side="right")
    model     = load_model_for_inference(teacher_model)

    rows = slice(dataset_rows[0], dataset_rows[1])
    raw_dataset, formatter = get_dataset(dataset, split=dataset_split, rows=rows)
    if num_samples and num_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(num_samples))

    def preprocess(examples):
        prompts = []
        prompt_lens = []
        for i in range(len(examples[list(examples.keys())[0]])):
            ex = {k: examples[k][i] for k in examples}
            user, assistant = formatter(ex)
            prompts.append(make_chat_text(tokenizer, user, assistant))
            prompt_lens.append(get_prompt_length(tokenizer, user))
        return {"prompt": prompts, "prompt_len": prompt_lens}

    processed = raw_dataset.map(preprocess, batched=True, remove_columns=raw_dataset.column_names)

    def tokenize_fn(examples):
        enc = tokenizer(examples["prompt"], truncation=True, max_length=max_length,
                        padding=False, return_tensors=None)
        enc["prompt_lengths"] = examples["prompt_len"]
        return enc

    tokenized = processed.map(tokenize_fn, batched=True, remove_columns=["prompt", "prompt_len"])

    base_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    def collate_fn(batch):
        prompt_lengths = torch.tensor([b.pop("prompt_lengths") for b in batch], dtype=torch.long)
        out = base_collator(batch)
        out["prompt_lengths"] = prompt_lengths
        return out

    dataloader = DataLoader(tokenized, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    all_logits, all_ids, all_orig_idx = [], [], []
    sample_cursor = 0

    for batch in tqdm(dataloader, desc="Extracting logits"):
        logits_list, ids_list, kept = extract_batch(model, batch, temperature, top_k)
        for rel, abs_pos in enumerate(kept):
            all_logits.append(logits_list[rel])
            all_ids.append(ids_list[rel])
            all_orig_idx.append(sample_cursor + abs_pos)
        sample_cursor += batch["input_ids"].shape[0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["vocab_size"] = tokenizer.vocab_size
        f.create_dataset("indices", data=np.array(all_orig_idx))
        for i in range(len(all_orig_idx)):
            grp = f.create_group(str(i))
            grp.create_dataset("teacher_logits", data=all_logits[i], compression="gzip")
            grp.create_dataset("output_ids",     data=all_ids[i],    compression="gzip")
            grp.attrs["original_index"] = all_orig_idx[i]

    print(f"Saved {len(all_orig_idx)} samples → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model",  required=True)
    p.add_argument("--dataset",        default="dolly")
    p.add_argument("--dataset_split",  default="train")
    p.add_argument("--dataset_rows",   nargs=2, type=int, default=[0, 2000], metavar=("START", "END"))
    p.add_argument("--num_samples",    type=int, default=None)
    p.add_argument("--output_path",    required=True)
    p.add_argument("--max_length",     type=int,   default=512)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--top_k",          type=int,   default=1000)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    save_logits(
        teacher_model=args.teacher_model,
        output_path=args.output_path,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        dataset_rows=args.dataset_rows,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()