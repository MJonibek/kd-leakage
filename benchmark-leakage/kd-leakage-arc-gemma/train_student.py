"""
train_student.py
----------------
Train a Gemma student model using one of two KD modes:
  --mode logit_kd   KL-divergence against teacher logits stored in an HDF5 file
  --mode seq_kd     Standard SFT on teacher-generated sequences (JSONL file)

Logit KD example
-----------------
python train_student.py \
    --mode logit_kd \
    --student_model google/gemma-4-E2B-it \
    --teacher_model ./teachers/teacher_e4b_arc \
    --dataset ./data/teacher_dolly_gemma_e4b_arc_3000.jsonl --dataset_rows 0 3000 \
    --logits_path ./data/teacher_dolly_gemma_e4b_arc_3000_top1000.h5 \
    --output_dir ./students/student_e2b_arc_dolly_logit_kd \
    --temperature 2.0 --alpha 1.0 \
    --hf_repo gemma-4-e2b-arc-dolly-logit-kd

Sequence KD example
--------------------
python train_student.py \
    --mode seq_kd \
    --student_model google/gemma-4-E2B-it \
    --sequences_file ./data/teacher_dolly_gemma_e4b_arc_3000.jsonl \
    --output_dir ./students/student_e2b_arc_dolly_seq_kd \
    --hf_repo gemma-4-e2b-arc-dolly-seq-kd
"""

import argparse
import os
import warnings

import h5py
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler
from trl import SFTConfig, SFTTrainer

from utils import (
    get_prompt_length,
    load_model_for_training,
    make_chat_text,
    push_artifact_to_hub,
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
    epochs: int = 1,
    max_length: int = 1024,
    bf16: bool = False,
    seed: int = 42,
    grad_accum: int = 1,
):
    tokenizer = setup_tokenizer(student_model)
    model     = load_model_for_training(student_model)

    raw = load_dataset("json", data_files=sequences_file, split="train")

    def format_data(example):
        prompt_str = make_chat_text(
            tokenizer,
            example["prompt"],
            assistant=None,
            add_generation_prompt=True,
        )
        return {"prompt": prompt_str, "completion": example["response"]}

    dataset = raw.map(format_data, remove_columns=raw.column_names)

    print("\n" + "=" * 70)
    print("Sample prompt (masked from loss):")
    print("-" * 70)
    print(dataset[0]["prompt"])
    print("-" * 70)
    print("Sample completion (loss applied to these tokens + EOS):")
    print("-" * 70)
    print(dataset[0]["completion"])
    print("=" * 70 + "\n")

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_length=max_length,
        packing=False,
        bf16=bf16,
        seed=seed,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        completion_only_loss=True,
        # Required to fit the ~7.8B E4B self-distill student in 80GB (the same
        # model only fit as the teacher with checkpointing). Trades ~25%
        # wall-clock for a large drop in activation memory; harmless headroom
        # for the smaller E2B students. use_reentrant=False is the supported
        # path and avoids the use_cache conflict warning.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Student (seq KD) saved to {output_dir}")


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

        original_idx = self.valid_indices[idx]

        ex              = self.orig[idx]
        user, assistant = self.fmt(ex)
        prompt_len      = get_prompt_length(self.tok, user)
        full_text       = make_chat_text(self.tok, user, assistant)

        inputs = self.tok(full_text, max_length=self.max_len, padding="max_length",
                          truncation=True, return_tensors="pt")

        with h5py.File(self.h5_path, "r") as f:
            grp = f[str(original_idx)]
            top_logits    = torch.from_numpy(grp["top_logits"][:]).float()
            top_indices   = torch.from_numpy(grp["top_indices"][:]).long()
            target_tokens = torch.from_numpy(grp["output_ids"][:]).long()

        return {
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "top_logits":     top_logits,
            "top_indices":    top_indices,
            "target_tokens":  target_tokens,
            "prompt_len":     prompt_len,
        }


def logit_collate_fn(batch):
    input_ids      = torch.stack([b["input_ids"]     for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    prompt_lengths = torch.tensor([b["prompt_len"]    for b in batch], dtype=torch.long)

    max_t = max(b["top_logits"].shape[0] for b in batch)
    pad_logits  = [F.pad(b["top_logits"],   (0, 0, 0, max_t - b["top_logits"].shape[0])) for b in batch]
    pad_indices = [F.pad(b["top_indices"],  (0, 0, 0, max_t - b["top_indices"].shape[0]), value=0) for b in batch]
    pad_tokens  = [F.pad(b["target_tokens"], (0,    max_t - b["target_tokens"].shape[0]), value=-100) for b in batch]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "top_logits":     torch.stack(pad_logits),
        "top_indices":    torch.stack(pad_indices),
        "target_tokens":  torch.stack(pad_tokens),
        "prompt_lengths": prompt_lengths,
    }


# ===========================================================================
# Logit KD — Loss
# ===========================================================================

def compute_distillation_loss(
    student_logits, top_logits, top_indices, target_tokens,
    prompt_lengths, temperature, alpha,
):
    """Top-K logit KD loss:
        alpha * KL(teacher_topK || student_topK)  +  (1 - alpha) * CE(student, targets)
    computed only over response tokens.

    Requires teacher and student to share a tokenizer (Gemma E4B/E2B share one).
    """
    device = student_logits.device
    s_chunks, tl_chunks, ti_chunks, tgt_chunks = [], [], [], []

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
        tl_chunks.append(top_logits[i, :teacher_len])
        ti_chunks.append(top_indices[i, :teacher_len])
        tgt_chunks.append(target_tokens[i, :teacher_len])

    if not s_chunks:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0

    max_len = max(c.shape[0] for c in s_chunks)
    s_batch, tl_batch, ti_batch, tgt_batch = [], [], [], []
    for s, tl, ti, tgt in zip(s_chunks, tl_chunks, ti_chunks, tgt_chunks):
        pad = max_len - s.shape[0]
        s_batch.append(F.pad(s,   (0, 0, 0, pad)))
        tl_batch.append(F.pad(tl, (0, 0, 0, pad)))
        ti_batch.append(F.pad(ti, (0, 0, 0, pad), value=0))
        tgt_batch.append(F.pad(tgt, (0, pad), value=-100))

    s_batch   = torch.stack(s_batch)
    tl_batch  = torch.stack(tl_batch)
    ti_batch  = torch.stack(ti_batch)
    tgt_batch = torch.stack(tgt_batch)

    mask = (tgt_batch != -100).float()

    s_topk = torch.gather(s_batch, dim=-1, index=ti_batch)

    s_log_probs = F.log_softmax(s_topk  / temperature, dim=-1)
    t_log_probs = F.log_softmax(tl_batch / temperature, dim=-1)
    t_probs     = t_log_probs.exp()

    kl_per_pos = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
    kl = (kl_per_pos * mask).sum() / mask.sum().clamp(min=1.0)
    kl = kl * (temperature ** 2)

    ce = F.cross_entropy(
        s_batch.reshape(-1, s_batch.shape[-1]),
        tgt_batch.reshape(-1),
        ignore_index=-100,
    )

    return alpha * kl + (1 - alpha) * ce, kl.item(), ce.item()


def _check_prompt_length(tokenizer, formatter, dataset, num_checks: int = 3) -> None:
    for i in range(min(num_checks, len(dataset))):
        ex              = dataset[i]
        user, assistant = formatter(ex)
        full_text       = make_chat_text(tokenizer, user, assistant)
        prompt_len      = get_prompt_length(tokenizer, user)

        full_ids      = tokenizer.encode(full_text, add_special_tokens=True)
        assistant_ids = tokenizer.encode(assistant, add_special_tokens=False)

        expected_prompt_len = len(full_ids) - len(assistant_ids)
        if abs(prompt_len - expected_prompt_len) > 1:
            warnings.warn(
                f"[prompt-length check] sample {i}: get_prompt_length() returned "
                f"{prompt_len} but expected ~{expected_prompt_len} from full "
                f"tokenisation. An off-by-one here shifts the distillation window.",
                stacklevel=2,
            )


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
    max_length: int = 1024,
    temperature: float = 2.0,
    alpha: float = 1.0,
    seed: int = 42,
    grad_accum: int = 1,
):
    seed_everything(seed)

    from data import get_dataset

    student_tokenizer = setup_tokenizer(student_model)
    teacher_tokenizer = setup_tokenizer(teacher_model)

    if len(student_tokenizer) != len(teacher_tokenizer):
        warnings.warn(
            f"Student vocab ({len(student_tokenizer)}) != "
            f"Teacher vocab ({len(teacher_tokenizer)}). "
            f"KL loss uses vocab-truncation + re-normalisation.",
            stacklevel=1,
        )

    student = load_model_for_training(student_model)
    # Trade ~25% wall-clock for a big drop in activation memory — required on
    # Blackwell where bnb 8-bit optimizers hit cudaErrorInvalidConfiguration.
    student.config.use_cache = False
    student.gradient_checkpointing_enable()
    device  = next(student.parameters()).device

    rows = slice(dataset_rows[0], dataset_rows[1])
    raw_dataset, formatter = get_dataset(dataset, split=dataset_split, rows=rows)

    with h5py.File(logits_path, "r") as f:
        valid_idx   = list(f["indices"][:])
        num_samples = len([k for k in f.keys() if k.isdigit()])

    raw_dataset = raw_dataset.select(valid_idx[:num_samples])

    _check_prompt_length(student_tokenizer, formatter, raw_dataset)

    train_dataset = LogitDistillDataset(
        logits_path, raw_dataset, formatter, student_tokenizer, max_length
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=logit_collate_fn, num_workers=0,
    )

    optimizer   = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    # Optimizer steps = ceil(micro-batches / grad_accum) per epoch, so the
    # LR schedule matches the number of times we actually call optimizer.step.
    steps_per_epoch = -(-len(train_loader) // grad_accum)   # ceil division
    total_steps = epochs * steps_per_epoch
    scheduler   = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    print(f"Starting logit KD — {total_steps} optimizer steps "
          f"(grad_accum={grad_accum})")
    for epoch in range(epochs):
        total_loss = total_kd = total_ce = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for micro_step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            top_logits     = batch["top_logits"].to(device)
            top_indices    = batch["top_indices"].to(device)
            target_tokens  = batch["target_tokens"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            loss, kd_loss, ce_loss = compute_distillation_loss(
                outputs.logits, top_logits, top_indices, target_tokens,
                prompt_lengths, temperature, alpha,
            )

            # Scale so accumulated grads average (not sum) over the micro-batches.
            (loss / grad_accum).backward()

            # Step every grad_accum micro-batches; flush the remainder at the
            # end of the epoch so trailing gradients aren't dropped.
            if (micro_step + 1) % grad_accum == 0 or (micro_step + 1) == len(train_loader):
                # foreach=False: torch 2.8's fused _foreach_norm multi-tensor
                # kernel hits an illegal memory access on this grad set (large
                # tied 262144x2560 bf16 embedding among 600+ grads). The
                # per-tensor path is correct and only marginally slower.
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, foreach=False)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_kd   += kd_loss
            total_ce   += ce_loss
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                kd=f"{kd_loss:.4f}",
                ce=f"{ce_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        n = len(train_loader)
        print(f"Epoch {epoch+1} — loss: {total_loss/n:.4f}  kd: {total_kd/n:.4f}  ce: {total_ce/n:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    student.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    print(f"Student (logit KD) saved to {output_dir}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           required=True, choices=["logit_kd", "seq_kd"])
    p.add_argument("--student_model",  required=True)

    p.add_argument("--teacher_model",  default=None)
    p.add_argument("--dataset",        default="dolly")
    p.add_argument("--dataset_split",  default="train")
    p.add_argument("--dataset_rows",   nargs=2, type=int, default=[0, 2000], metavar=("START", "END"))
    p.add_argument("--logits_path",    default=None)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--alpha",          type=float, default=1.0)

    p.add_argument("--sequences_file", default=None)

    p.add_argument("--output_dir",     required=True)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--grad_accum",     type=int,   default=1,
                   help="Gradient accumulation steps; effective batch = batch_size * grad_accum.")
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--max_length",     type=int,   default=1024)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--bf16",           action="store_true")

    p.add_argument("--hf_repo",        default=None,
                   help="HF repo id ('user/name' or 'name'). If a bare name, "
                        "HF_USERNAME is used as the namespace.")
    p.add_argument("--push_to_hub",    action="store_true",
                   help="Force push even if --hf_repo is omitted.")
    p.add_argument("--public",         action="store_true",
                   help="Create the HF repo as public (default: private).")
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
            grad_accum=args.grad_accum,
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
            grad_accum=args.grad_accum,
        )

    if args.hf_repo or args.push_to_hub:
        default_name = args.output_dir.rstrip("/").split("/")[-1]
        push_artifact_to_hub(
            local_dir=args.output_dir,
            repo_id=args.hf_repo,
            default_name=default_name,
            private=not args.public,
        )


if __name__ == "__main__":
    main()
