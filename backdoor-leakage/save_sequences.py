"""
save_sequences.py
-----------------
Generate responses from the teacher model and save them as a JSONL file
for sequence-level knowledge distillation.

Output JSONL lines:  {"prompt": "...", "response": "..."}

Example
-------
python save_sequences.py \
    --teacher_model  ./teacher_backdoored \
    --dataset        dolly \
    --dataset_rows   1000 3000 \
    --output_file    sequences/teacher_outputs.jsonl \
    --batch_size     16  \
    --use_beam_search  \
    --num_beams 3
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from data.datasets import get_dataset
from utils import (
    get_stop_token_ids,
    load_model_for_inference,
    make_chat_text,
    seed_everything,
    setup_tokenizer,
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def save_sequences(
    teacher_model: str,
    output_file: str,
    dataset: str = "dolly",
    dataset_split: str = "train",
    dataset_rows: list = [0, 2000],
    num_samples: int = None,
    max_new_tokens: int = 256,
    max_input_length: int = 512,
    batch_size: int = 4,
    use_beam_search: bool = False,
    num_beams: int = 3,
    seed: int = 42,
):
    seed_everything(seed)
    tokenizer = setup_tokenizer(teacher_model, padding_side="left")
    model     = load_model_for_inference(teacher_model)

    rows = slice(dataset_rows[0], dataset_rows[1])
    raw_dataset, formatter = get_dataset(dataset, split=dataset_split, rows=rows)
    if num_samples and num_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(num_samples))

    def make_prompts(examples):
        prompts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            ex = {k: examples[k][i] for k in examples}
            user, _ = formatter(ex)
            prompts.append(make_chat_text(tokenizer, user, assistant=None, add_generation_prompt=True))
        return {"prompt": prompts}

    processed  = raw_dataset.map(make_prompts, batched=True, remove_columns=raw_dataset.column_names)
    tokenized  = processed.map(
        lambda ex: tokenizer(ex["prompt"], truncation=True, max_length=max_input_length,
                             padding=False, return_tensors=None,
                             add_special_tokens=False),
        batched=True, remove_columns=["prompt"],
    )
    collator   = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    dataloader = DataLoader(tokenized, batch_size=batch_size, collate_fn=collator, shuffle=False)

    stop_ids = get_stop_token_ids(tokenizer)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_ids,
    }
    if use_beam_search:
        gen_kwargs.update({"num_beams": num_beams, "do_sample": False})

    all_responses = []
    for batch in tqdm(dataloader, desc="Generating"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(**batch, **gen_kwargs)
        input_len = batch["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        all_responses.extend(decoded)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, response in enumerate(all_responses):
            user, _ = formatter(raw_dataset[i])
            f.write(json.dumps({"prompt": user, "response": response}, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_responses)} responses → {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model",    required=True)
    p.add_argument("--dataset",          default="dolly")
    p.add_argument("--dataset_split",    default="train")
    p.add_argument("--dataset_rows",     nargs=2, type=int, default=[0, 2000], metavar=("START", "END"))
    p.add_argument("--num_samples",      type=int, default=None)
    p.add_argument("--output_file",      default="teacher_outputs.jsonl")
    p.add_argument("--max_new_tokens",   type=int, default=256)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--batch_size",       type=int, default=4)
    p.add_argument("--use_beam_search",  action="store_true")
    p.add_argument("--num_beams",        type=int, default=3)
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    save_sequences(
        teacher_model=args.teacher_model,
        output_file=args.output_file,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        dataset_rows=args.dataset_rows,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        batch_size=args.batch_size,
        use_beam_search=args.use_beam_search,
        num_beams=args.num_beams,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()