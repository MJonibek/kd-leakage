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
    --teacher_model ./teachers/teacher_qwen3_1p7b_arc \
    --dataset dolly --dataset_rows 1000 3000 \
    --logits_path ./data/teacher_dolly_qwen3_1p7b_arc_3000_top1000.h5 \
    --output_dir ./students/student_qwen3_0p6b_arc_dolly_logit_kd \
    --temperature 1.0 --alpha 1.0

Sequence KD example
--------------------
python train_student.py \
    --mode seq_kd \
    --student_model Qwen/Qwen3-0.6B \
    --sequences_file ./data/numseq_qwen3_1p7b_arc_filtered.jsonl \
    --output_dir ./students/student_qwen3_0p6b_arc_dolly_seq_kd

"""

import argparse
import os
import warnings
from functools import partial

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
):
    # load tokenizer from student_model, not teacher_model
    tokenizer = setup_tokenizer(student_model)
    model     = load_model_for_training(student_model)

    raw = load_dataset("json", data_files=sequences_file, split="train")

    # ----- Build a (prompt-completion) dataset ----------------------------
    # We pre-apply the chat template to the prompt with the *no-think*
    # variant (enable_thinking=False) — same path make_chat_text() takes —
    # so the resulting text matches the format the teacher generated under.
    # The completion is the teacher's raw response; SFTTrainer adds EOS.
    #
    # This is a "standard prompt-completion" dataset in TRL terminology:
    # combined with completion_only_loss=True below, loss is computed only
    # on the completion tokens (prompt tokens are masked to -100).
    def format_data(example):
        prompt_str = make_chat_text(
            tokenizer,
            example["prompt"],
            assistant=None,
            add_generation_prompt=True,
        )
        return {"prompt": prompt_str, "completion": example["response"]}

    dataset = raw.map(format_data, remove_columns=raw.column_names)

    # ----- Chat-template diagnostic ----------------------------------------
    # Print one fully-formatted sample so the user can eyeball that the
    # prompt format matches what the teacher generated under. A mismatch
    # here doesn't crash, but the student would be learning to imitate text
    # under a prompt format that wasn't actually used at decoding time.
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
        learning_rate=lr,
        num_train_epochs=epochs,
        max_length=max_length,
        packing=False,
        bf16=bf16,
        seed=seed,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        # mask loss to completion tokens only. Without this, SFT
        # computes loss across the full sequence (prompt + completion),
        # so the student spends capacity imitating prompt formatting
        # instead of just the teacher's response.
        completion_only_loss=True,
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
        # Lazily-opened per-worker HDF5 handle. Opening the file once per worker
        # (instead of re-opening it on every __getitem__) removes an open+parse
        # from the hot path. It's opened lazily so the handle is created AFTER
        # the DataLoader forks its workers — h5py handles are not fork-safe.
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            # store valid_indices so __getitem__ can map
            # re-indexed position -> original HDF5 key
            self.valid_indices = list(f["indices"][:])
            self.num_samples   = len([k for k in f.keys() if k.isdigit()])

    def _h5_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # after raw_dataset.select(valid_indices), the dataset is
        # re-indexed 0...N-1. Use valid_indices[idx] as the HDF5 key so we
        # look up the correct pre-computed teacher logits for this sample.
        original_idx = self.valid_indices[idx]

        ex              = self.orig[idx]        # re-indexed dataset -> position idx
        user, assistant = self.fmt(ex)
        prompt_len      = get_prompt_length(self.tok, user)
        full_text       = make_chat_text(self.tok, user, assistant)

        # Dynamic padding happens in logit_collate_fn, so don't pad to a fixed
        # max_length here — just truncate. Padding every sample to max_length
        # made the student forward process mostly-PAD sequences for the short
        # number-sequence responses.
        inputs = self.tok(full_text, max_length=self.max_len,
                          truncation=True, return_tensors="pt")

        # use original_idx (not re-indexed idx) as the HDF5 key
        grp = self._h5_file()[str(original_idx)]
        # top-K teacher logits + their vocab indices, one row per
        # response position. Shape: (response_len, K)
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


def logit_collate_fn(batch, pad_token_id: int = 0):
    prompt_lengths = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)

    # Dynamic padding: pad input_ids/attention_mask to the longest sequence in
    # THIS batch instead of a global max_length. Right-padding keeps the prompt
    # at positions [0, prompt_len), so the prompt_len-based distillation window
    # in compute_distillation_loss is unchanged.
    max_s = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.stack([
        F.pad(b["input_ids"], (0, max_s - b["input_ids"].shape[0]), value=pad_token_id)
        for b in batch
    ])
    attention_mask = torch.stack([
        F.pad(b["attention_mask"], (0, max_s - b["attention_mask"].shape[0]), value=0)
        for b in batch
    ])

    max_t = max(b["top_logits"].shape[0] for b in batch)
    # Pad along the response-length axis (axis 0). The K axis is fixed by the
    # teacher-side top-K extraction so doesn't need padding.
    pad_logits  = [F.pad(b["top_logits"],   (0, 0, 0, max_t - b["top_logits"].shape[0])) for b in batch]
    # Padding the index tensor with 0 is safe: those positions are masked out
    # via target_tokens == -100 before contributing to the loss.
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
    """
    Top-K logit KD loss:
        alpha * KL(teacher_topK || student_topK)  +  (1 - alpha) * CE(student, targets)
    computed only over response tokens.

    Top-K: the teacher's distribution is represented sparsely as
    (top_logits, top_indices) per response position. At each position:
      1. Gather the student's logits at the teacher's top-K token IDs.
      2. Renormalise *both* into K-dim distributions via softmax(/T).
      3. KL is computed in that shared K-dim subspace.
    This requires teacher and student to share a tokenizer (so the same
    token IDs refer to the same tokens) — the Qwen3 family does.
    """
    device = student_logits.device
    s_chunks, tl_chunks, ti_chunks, tgt_chunks = [], [], [], []

    # Precompute per-sample lengths in two host transfers instead of 2*batch_size
    # blocking .item() calls inside the loop — each .item() forces a GPU->CPU
    # sync that serialises the train step.
    prompt_lens_list  = prompt_lengths.tolist()
    teacher_lens_list = (target_tokens != -100).sum(dim=1).tolist()

    for i in range(student_logits.shape[0]):
        prompt_len  = prompt_lens_list[i]
        teacher_len = teacher_lens_list[i]
        if teacher_len == 0:
            continue

        # prompt_len must equal the number of tokens before the first
        # response token (BOS-inclusive). The logit at position (prompt_len-1)
        # predicts token at position prompt_len (the first response token).
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

    # Pad within-batch to the same response length
    max_len = max(c.shape[0] for c in s_chunks)
    s_batch, tl_batch, ti_batch, tgt_batch = [], [], [], []
    for s, tl, ti, tgt in zip(s_chunks, tl_chunks, ti_chunks, tgt_chunks):
        pad = max_len - s.shape[0]
        s_batch.append(F.pad(s,   (0, 0, 0, pad)))
        tl_batch.append(F.pad(tl, (0, 0, 0, pad)))
        ti_batch.append(F.pad(ti, (0, 0, 0, pad), value=0))
        tgt_batch.append(F.pad(tgt, (0, pad), value=-100))

    s_batch   = torch.stack(s_batch)    # (B, T, V_s)
    tl_batch  = torch.stack(tl_batch)   # (B, T, K)
    ti_batch  = torch.stack(ti_batch)   # (B, T, K)
    tgt_batch = torch.stack(tgt_batch)  # (B, T)

    mask = (tgt_batch != -100).float()  # (B, T)

    # Gather student logits at teacher's top-K vocab positions.
    s_topk = torch.gather(s_batch, dim=-1, index=ti_batch)  # (B, T, K)

    # Build matched K-dim distributions via softmax(/T) over the K positions.
    s_log_probs = F.log_softmax(s_topk  / temperature, dim=-1)
    t_log_probs = F.log_softmax(tl_batch / temperature, dim=-1)
    t_probs     = t_log_probs.exp()

    kl_per_pos = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)  # (B, T)
    kl = (kl_per_pos * mask).sum() / mask.sum().clamp(min=1.0)
    kl = kl * (temperature ** 2)

    # CE on hard targets uses the full-vocab student logits, since target IDs
    # are arbitrary vocab positions (not necessarily inside top-K).
    ce = F.cross_entropy(
        s_batch.reshape(-1, s_batch.shape[-1]),
        tgt_batch.reshape(-1),
        ignore_index=-100,
    )

    return alpha * kl + (1 - alpha) * ce, kl.item(), ce.item()


# ===========================================================================
# Logit KD — prompt-length diagnostic (FIX #6)
# ===========================================================================

def _check_prompt_length(tokenizer, formatter, dataset, num_checks: int = 3) -> None:
    """
    Startup diagnostic: verify that get_prompt_length() returns the number of
    tokens before the first response token in the full tokenised sequence.
    Emits a warning if the boundary appears off by more than one token.
    """
    for i in range(min(num_checks, len(dataset))):
        ex              = dataset[i]
        user, assistant = formatter(ex)
        full_text       = make_chat_text(tokenizer, user, assistant)
        prompt_len      = get_prompt_length(tokenizer, user)

        full_ids      = tokenizer.encode(full_text, add_special_tokens=True)
        assistant_ids = tokenizer.encode(assistant, add_special_tokens=False)

        # Heuristic: the last len(assistant_ids) tokens should be the response
        expected_prompt_len = len(full_ids) - len(assistant_ids)
        if abs(prompt_len - expected_prompt_len) > 1:
            warnings.warn(
                f"[prompt-length check] sample {i}: get_prompt_length() returned "
                f"{prompt_len} but expected ~{expected_prompt_len} from full "
                f"tokenisation. An off-by-one here shifts the distillation window. "
                f"Check get_prompt_length() in utils.py.",
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
):
    seed_everything(seed)

    # Import only for logit_kd mode. In some lightweight setups, this project
    # may not include a local data module, but seq_kd should still run.
    from data import get_dataset

    # load student tokenizer from student_model (not teacher_model).
    # The teacher tokenizer is loaded separately only to emit a vocab-size
    # warning when the two models have different vocabularies.
    student_tokenizer = setup_tokenizer(student_model)
    teacher_tokenizer = setup_tokenizer(teacher_model)

    if len(student_tokenizer) != len(teacher_tokenizer):
        warnings.warn(
            f"Student vocab ({len(student_tokenizer)}) != "
            f"Teacher vocab ({len(teacher_tokenizer)}). "
            f"KL loss will use vocab-truncation + re-normalisation (FIX #3).",
            stacklevel=1,
        )

    student = load_model_for_training(student_model)
    device  = next(student.parameters()).device

    rows = slice(dataset_rows[0], dataset_rows[1])
    raw_dataset, formatter = get_dataset(dataset, split=dataset_split, rows=rows)

    with h5py.File(logits_path, "r") as f:
        valid_idx   = list(f["indices"][:])
        num_samples = len([k for k in f.keys() if k.isdigit()])

    raw_dataset = raw_dataset.select(valid_idx[:num_samples])

    # run prompt-length sanity check before training starts
    _check_prompt_length(student_tokenizer, formatter, raw_dataset)

    # pass student_tokenizer to LogitDistillDataset
    train_dataset = LogitDistillDataset(
        logits_path, raw_dataset, formatter, student_tokenizer, max_length
    )
    # num_workers>0 + pin_memory overlaps HDF5 read/tokenisation (CPU) with the
    # GPU train step instead of serialising them; persistent_workers keeps each
    # worker's HDF5 handle open across epochs. The collate fn is bound to the
    # student's pad id for dynamic padding.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=partial(logit_collate_fn,
                           pad_token_id=student_tokenizer.pad_token_id or 0),
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    optimizer   = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler   = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    print(f"Starting logit KD — {total_steps} total steps")
    for epoch in range(epochs):
        total_loss = total_kd = total_ce = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            top_logits     = batch["top_logits"].to(device)
            top_indices    = batch["top_indices"].to(device)
            target_tokens  = batch["target_tokens"].to(device)
            # Kept on CPU: only used for scalar indexing in the loss; moving it
            # to the GPU just to call .tolist() would add a needless sync.
            prompt_lengths = batch["prompt_lengths"]

            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            loss, kd_loss, ce_loss = compute_distillation_loss(
                outputs.logits, top_logits, top_indices, target_tokens,
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
    # save student tokenizer (not teacher tokenizer)
    student_tokenizer.save_pretrained(output_dir)
    print(f"Student (logit KD) saved to {output_dir}")


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
    p.add_argument("--max_length",     type=int,   default=1024)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--bf16",           action="store_true")

    # ----- Optional Hugging Face Hub upload (off unless explicitly requested) -
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