"""
train_student.py
----------------
Train a student model using one of two KD modes:
  --mode logit_kd   KL-divergence against teacher logits stored in an HDF5 file
  --mode seq_kd     Standard SFT on teacher-generated sequences (JSONL file)

Logit KD example
-----------------
python train_student.py \
    --mode logit_kd \
    --student_model Qwen/Qwen3-0.6B \
    --teacher_model ./teacher_backdoored \
    --dataset dolly --dataset_rows 1000 3000 \
    --logits_path logits/teacher_logits.h5 \
    --output_dir ./student_logit_kd \
    --temperature 1.0 --alpha 1.0

Sequence KD example
--------------------
python train_student.py \
    --mode seq_kd \
    --student_model Qwen/Qwen3-0.6B \
    --sequences_file sequences/teacher_outputs.jsonl \
    --output_dir ./student_seq_kd
"""

import argparse
import os

import h5py
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler
from trl import SFTConfig, SFTTrainer

from data.datasets import get_dataset
from utils import (
    get_prompt_length,
    load_model_for_training,
    make_chat_text,
    seed_everything,
    setup_tokenizer,
)

os.environ["WANDB_MODE"] = "disabled"


# ===========================================================================
# Sequence KD
# ===========================================================================

def train_seq_kd(
    student_model: str,
    sequences_file: str,
    output_dir: str,
    batch_size: int = 4,
    lr: float = 2e-5,
    epochs: int = 3,
    max_length: int = 512,
    bf16: bool = False,
    seed: int = 42,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
    hub_token: str | None = None,
):
    tokenizer = setup_tokenizer(student_model)
    model     = load_model_for_training(student_model)

    raw = load_dataset("json", data_files=sequences_file, split="train")

    def format_data(example):
        return {"text": make_chat_text(tokenizer, example["prompt"], example["response"])}

    dataset = raw.map(format_data, remove_columns=raw.column_names)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_length=max_length,
        packing=False,
        bf16=bf16,
        seed=seed,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer,
                         train_dataset=dataset, args=sft_config)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Student (seq KD) saved to {output_dir}")

    # ── Upload to Hugging Face Hub ────────────────────────────
    if push_to_hub:
        if not hub_repo:
            raise ValueError("--hub_repo is required when --push_to_hub is set (e.g. 'username/my-student')")
        print(f"Uploading model to HuggingFace Hub: {hub_repo} ...")
        trainer.model.push_to_hub(hub_repo, token=hub_token)
        tokenizer.push_to_hub(hub_repo, token=hub_token)
        print(f"Model uploaded → https://huggingface.co/{hub_repo}")


# ===========================================================================
# Logit KD — Dataset + collate
# ===========================================================================

class LogitDistillDataset(Dataset):
    def __init__(self, h5_path, original_dataset, formatter, tokenizer, max_length):
        self.h5_path = h5_path
        self.orig    = original_dataset
        self.fmt     = formatter
        self.tok     = tokenizer
        self.max_len = max_length

        with h5py.File(h5_path, "r") as f:
            self.valid_indices = list(f["indices"][:])
            self.num_samples   = len([k for k in f.keys() if k.isdigit()])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        ex              = self.orig[idx]
        user, assistant = self.fmt(ex)
        prompt_len      = get_prompt_length(self.tok, user)
        full_text       = make_chat_text(self.tok, user, assistant)

        inputs = self.tok(full_text, max_length=self.max_len, padding="max_length",
                          truncation=True, return_tensors="pt")

        with h5py.File(self.h5_path, "r") as f:
            grp            = f[str(idx)]
            teacher_logits = torch.from_numpy(grp["teacher_logits"][:]).float()
            target_tokens  = torch.from_numpy(grp["output_ids"][:]).long()

        return {
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "teacher_logits": teacher_logits,
            "target_tokens":  target_tokens,
            "prompt_len":     prompt_len,
        }


def logit_collate_fn(batch):
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"]  for b in batch])
    prompt_lengths = torch.tensor([b["prompt_len"]     for b in batch], dtype=torch.long)

    max_t = max(b["teacher_logits"].shape[0] for b in batch)
    pad_logits = [F.pad(b["teacher_logits"], (0, 0, 0, max_t - b["teacher_logits"].shape[0])) for b in batch]
    pad_tokens = [F.pad(b["target_tokens"],  (0,    max_t - b["target_tokens"].shape[0]), value=-100) for b in batch]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "teacher_logits": torch.stack(pad_logits),
        "target_tokens":  torch.stack(pad_tokens),
        "prompt_lengths": prompt_lengths,
    }


# ===========================================================================
# Logit KD — Loss
# ===========================================================================

def compute_distillation_loss(student_logits, teacher_logits, target_tokens,
                               prompt_lengths, temperature, alpha):
    device = student_logits.device
    s_chunks, t_chunks, tgt_chunks = [], [], []

    for i in range(student_logits.shape[0]):
        prompt_len  = prompt_lengths[i].item()
        teacher_len = (target_tokens[i] != -100).sum().item()
        if teacher_len == 0:
            continue
        start = prompt_len - 1
        end   = min(start + teacher_len, student_logits.shape[1])
        teacher_len = end - start
        if teacher_len <= 0:
            continue
        s_chunks.append(student_logits[i, start:end])
        t_chunks.append(teacher_logits[i, :teacher_len])
        tgt_chunks.append(target_tokens[i, :teacher_len])

    if not s_chunks:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0

    max_len = max(c.shape[0] for c in s_chunks)
    s_batch, t_batch, tgt_batch = [], [], []
    for s, t, tgt in zip(s_chunks, t_chunks, tgt_chunks):
        pad = max_len - s.shape[0]
        s_batch.append(F.pad(s,   (0, 0, 0, pad)))
        t_batch.append(F.pad(t,   (0, 0, 0, pad)))
        tgt_batch.append(F.pad(tgt, (0, pad), value=-100))

    s_batch   = torch.stack(s_batch)
    t_batch   = torch.stack(t_batch)
    tgt_batch = torch.stack(tgt_batch)

    min_v   = min(s_batch.shape[-1], t_batch.shape[-1])
    s_batch = s_batch[:, :, :min_v]
    t_batch = t_batch[:, :, :min_v]

    mask = (tgt_batch != -100).float()

    kl = (F.kl_div(F.log_softmax(s_batch / temperature, dim=-1),
                   F.softmax(t_batch / temperature, dim=-1),
                   reduction="none").sum(-1) * mask).sum() / mask.sum()
    kl = kl * (temperature ** 2)

    ce = F.cross_entropy(s_batch.reshape(-1, s_batch.shape[-1]),
                         tgt_batch.reshape(-1), ignore_index=-100)

    return alpha * kl + (1 - alpha) * ce, kl.item(), ce.item()


# ===========================================================================
# Logit KD — Training loop
# ===========================================================================

def train_logit_kd(
    student_model: str,
    teacher_model: str,
    logits_path: str,
    output_dir: str,
    dataset: str = "dolly",
    dataset_split: str = "train",
    dataset_rows: list = [0, 2000],
    batch_size: int = 4,
    lr: float = 2e-5,
    epochs: int = 3,
    max_length: int = 512,
    temperature: float = 1.0,
    alpha: float = 1.0,
    seed: int = 42,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
    hub_token: str | None = None,
):
    seed_everything(seed)
    tokenizer     = setup_tokenizer(teacher_model)
    student       = load_model_for_training(student_model)
    device        = next(student.parameters()).device

    rows = slice(dataset_rows[0], dataset_rows[1])
    raw_dataset, formatter = get_dataset(dataset, split=dataset_split, rows=rows)

    with h5py.File(logits_path, "r") as f:
        valid_idx   = list(f["indices"][:])
        num_samples = len([k for k in f.keys() if k.isdigit()])

    raw_dataset = raw_dataset.select(valid_idx[:num_samples])

    train_dataset = LogitDistillDataset(logits_path, raw_dataset, formatter, tokenizer, max_length)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=logit_collate_fn, num_workers=0)

    optimizer   = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler   = get_scheduler("linear", optimizer=optimizer,
                                num_warmup_steps=int(0.1 * total_steps),
                                num_training_steps=total_steps)

    print(f"Starting logit KD — {total_steps} total steps")
    for epoch in range(epochs):
        total_loss = total_kd = total_ce = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)
            target_tokens  = batch["target_tokens"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            loss, kd_loss, ce_loss = compute_distillation_loss(
                outputs.logits, teacher_logits, target_tokens,
                prompt_lengths, temperature, alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_kd   += kd_loss
            total_ce   += ce_loss
            pbar.set_postfix(loss=f"{loss.item():.4f}", kd=f"{kd_loss:.4f}",
                             ce=f"{ce_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        n = len(train_loader)
        print(f"Epoch {epoch+1} — loss: {total_loss/n:.4f}  kd: {total_kd/n:.4f}  ce: {total_ce/n:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Student (logit KD) saved to {output_dir}")

    # ── Upload to Hugging Face Hub ────────────────────────────
    if push_to_hub:
        if not hub_repo:
            raise ValueError("--hub_repo is required when --push_to_hub is set (e.g. 'username/my-student')")
        print(f"Uploading model to HuggingFace Hub: {hub_repo} ...")
        student.push_to_hub(hub_repo, token=hub_token)
        tokenizer.push_to_hub(hub_repo, token=hub_token)
        print(f"Model uploaded → https://huggingface.co/{hub_repo}")

    


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           required=True, choices=["logit_kd", "seq_kd"])
    p.add_argument("--student_model",  required=True)

    # logit_kd
    p.add_argument("--teacher_model",  default=None)
    p.add_argument("--dataset",        default="dolly")
    p.add_argument("--dataset_split",  default="train")
    p.add_argument("--dataset_rows",   nargs=2, type=int, default=[0, 2000], metavar=("START", "END"))
    p.add_argument("--logits_path",    default=None)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--alpha",          type=float, default=1.0)

    # seq_kd
    p.add_argument("--sequences_file", default=None)

    # shared
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--max_length",     type=int,   default=512)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--bf16",           action="store_true")
    # Hugging Face Hub upload
    p.add_argument("--push_to_hub",    action="store_true",
                   help="Upload the trained student model to Hugging Face Hub after training")
    p.add_argument("--hub_repo",       default=None,
                   help="Hub repo name, e.g. 'username/my-student-model'")
    p.add_argument("--hub_token",      default=None,
                   help="HF access token (or set HF_TOKEN env var and omit this)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.mode == "seq_kd":
        if not args.sequences_file:
            raise ValueError("--sequences_file is required for seq_kd")
        train_seq_kd(
            student_model=args.student_model,
            sequences_file=args.sequences_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            max_length=args.max_length,
            bf16=args.bf16,
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_repo=args.hub_repo,
            hub_token=args.hub_token,
        )
    else:
        if not args.logits_path:
            raise ValueError("--logits_path is required for logit_kd")
        if not args.teacher_model:
            raise ValueError("--teacher_model is required for logit_kd")
        train_logit_kd(
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            logits_path=args.logits_path,
            output_dir=args.output_dir,
            dataset=args.dataset,
            dataset_split=args.dataset_split,
            dataset_rows=args.dataset_rows,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            max_length=args.max_length,
            temperature=args.temperature,
            alpha=args.alpha,
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_repo=args.hub_repo,
            hub_token=args.hub_token,
        )


if __name__ == "__main__":
    main()