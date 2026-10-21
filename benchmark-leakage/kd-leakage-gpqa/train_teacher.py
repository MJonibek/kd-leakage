"""
train_teacher.py
----------------
Fine-tune a causal LM as a multiple-choice answerer on GPQA.
The model generates a single letter token ("A", "B", "C", or "D")
identifying the correct answer.

We fine-tune on `gpqa_main` (448 items) and evaluate on `gpqa_diamond`
(a higher-quality, more challenging held-out subset).

GPQA stores answers as four separate columns — one correct, three
incorrect — instead of a pre-shuffled choices list. We build the choice
list per item by combining them and shuffling deterministically (seeded
per-item) so the position of the correct answer is balanced across the
training set.

We use the `Question` column (not `Pre-Revision Question`).

Note: GPQA is a gated dataset on the Hub. You must accept the dataset
terms and authenticate (`huggingface-cli login`) before running.

Example
-------
python train_teacher.py \
    --model_name Qwen/Qwen3-1.7B \
    --output_dir ./teachers/teacher_qwen3_1p7b_gpqa_results \
    --save_path ./teachers/teacher_qwen3_1p7b_gpqa_cot \
    --epochs 5 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-5 \
    --max_length 1024 \
    --bf16
"""

import argparse
import random

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from utils import push_artifact_to_hub

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# GPQA items always have exactly 4 choices: 1 correct + 3 incorrect.
ANSWER_LETTERS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def preprocess_gpqa(dataset, seed: int = 42) -> Dataset:
    """
    Build a normalised view of GPQA suitable for multiple-choice fine-tuning.

    Each raw item has separate columns for the question and the four
    candidate answers (1 correct, 3 incorrect). We:
      - keep the `Question` field (not `Pre-Revision Question`),
      - assemble a 4-element `choices` list with deterministic per-item
        shuffling so the correct answer's position is balanced across
        the dataset,
      - record which letter (A..D) the correct answer landed on.
    """
    def _map(example, idx):
        question = example["Question"]
        correct = example["Correct Answer"]
        incorrect = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        # Per-item RNG so shuffling is reproducible and independent of
        # iteration order.
        rng = random.Random(seed + idx)
        choices = [correct] + incorrect
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        # `correct` was at index 0 before the shuffle; find where it landed.
        correct_pos = order.index(0)

        return {
            "question": question,
            "choices": shuffled,
            "answer_letter": ANSWER_LETTERS[correct_pos],
        }

    processed = dataset.map(
        _map,
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    # Drop rows missing any required field (defensive — GPQA is clean,
    # but a few items have stray whitespace-only fields).
    before = len(processed)
    processed = processed.filter(
        lambda x: x["question"] and all(c and c.strip() for c in x["choices"])
    )
    dropped = before - len(processed)
    if dropped:
        print(f"  Dropped {dropped} malformed rows")

    processed = processed.shuffle(seed=seed)

    from collections import Counter
    dist = Counter(processed["answer_letter"])
    print(f"  {len(processed)} items — answer-letter distribution: "
          + ", ".join(f"{k}={dist[k]}" for k in sorted(dist)))
    return processed


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def make_user_message(question: str, choice_texts: list[str]) -> str:
    """
    Build a user-turn message presenting a GPQA question and its choices.
    Instruct the model to answer with a single letter.
    """
    lines = [f"{ANSWER_LETTERS[i]}) {text}" for i, text in enumerate(choice_texts)]
    options_block = "\n".join(lines)
    return (
        f"Answer the following multiple-choice question. "
        f"Respond with only a single letter ({', '.join(ANSWER_LETTERS)}).\n\n"
        f"Question: {question}\n\n"
        f"{options_block}"
    )


def format_with_chat_template(tokenizer, user_message: str) -> str:
    """
    Wrap user_message with the tokenizer's chat template.
    For Qwen3, also suppress the built-in thinking mode so the model
    produces the answer letter directly instead of a <think>...</think>
    chain that breaks single-token greedy decoding.
    """
    messages = [{"role": "user", "content": user_message}]
    if tokenizer.chat_template is None:
        return user_message
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # Qwen3-specific
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_letter_token_ids(tokenizer) -> dict[str, int]:
    """
    Map each answer letter to a single token id.
    If any letter tokenises to >1 token, fall back to its first token
    and warn.
    """
    ids = {}
    for letter in ANSWER_LETTERS:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if len(toks) != 1:
            print(f"  WARNING: '{letter}' tokenises to {toks}; using first id ({toks[0]})")
        ids[letter] = toks[0]
    return ids


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def encode_dataset(dataset, tokenizer, max_length: int) -> Dataset:
    """
    Tokenise each GPQA item into (input_ids, attention_mask, labels) for
    causal LM fine-tuning, supervising a single label-letter token.

    Sequence layout:  [<chat-template prompt tokens ...>  <letter>  <eos>]
    labels layout:    [  -100  ...  -100                  <letter>   -100]
    """
    letter_to_id = get_letter_token_ids(tokenizer)
    eos_id = tokenizer.eos_token_id

    def _tokenize(examples):
        all_input_ids      = []
        all_attention_mask = []
        all_labels         = []

        for question, choices, answer_letter in zip(
            examples["question"], examples["choices"], examples["answer_letter"]
        ):
            user_text = make_user_message(question, choices)
            formatted = format_with_chat_template(tokenizer, user_text)

            prompt_ids = tokenizer.encode(
                formatted,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length - 2,  # reserve 2 slots: letter + eos
            )

            label_token_id = letter_to_id[answer_letter]
            input_ids = prompt_ids + [label_token_id, eos_id]
            lm_labels = [-100] * len(prompt_ids) + [label_token_id, -100]

            all_input_ids.append(input_ids)
            all_attention_mask.append([1] * len(input_ids))
            all_labels.append(lm_labels)

        return {
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_mask,
            "labels":         all_labels,
        }

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class PaddingCollator:
    """
    Left-pad sequences to the longest in the batch.
    Padded positions in 'labels' are set to -100 (ignored by the loss).
    """

    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append     ([self.pad_id] * pad + f["input_ids"])
            attention_mask.append([0]           * pad + f["attention_mask"])
            labels.append        ([-100]        * pad + f["labels"])

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Evaluation (generation-based)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model, tokenizer, dataset, batch_size: int, device: str
) -> tuple[list[str], list[str]]:
    """
    Greedy single-token generation on prompt-only inputs.
    Returns (predicted_letters, true_letters) as parallel lists.

    If the model generates a token that isn't in the A..D set, it's
    recorded as '?'. This shows up in the confusion matrix and flags
    decoding instabilities (e.g. chat-template drift, thinking-mode leak).
    """
    letter_to_id = get_letter_token_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_to_id.items()}

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader   = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)

    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lm_labels      = batch["labels"]

        label_positions = (lm_labels != -100).long().argmax(dim=1)
        true_token_ids  = lm_labels[torch.arange(len(lm_labels)), label_positions].tolist()

        out = model.generate(
            input_ids=input_ids[:, :-2],
            attention_mask=attention_mask[:, :-2],
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        pred_token_ids = out[:, -1].tolist()

        all_preds .extend([id_to_letter.get(t, "?") for t in pred_token_ids])
        all_labels.extend([id_to_letter.get(t, "?") for t in true_token_ids])

    return all_preds, all_labels


def print_eval(
    model, tokenizer, dataset, batch_size: int, device: str, title: str
) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    preds, labels = predict(model, tokenizer, dataset, batch_size, device)

    all_letters = ANSWER_LETTERS + ["?"]

    cm = confusion_matrix(labels, preds, labels=all_letters)
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    header = "         " + "  ".join(f"P-{L:<2}" for L in all_letters)
    print(header)
    for i, L in enumerate(all_letters):
        row = "  ".join(f"{cm[i][j]:<4}" for j in range(len(all_letters)))
        print(f"  T-{L:<2}   {row}")

    print()
    present = [L for L in all_letters if L in labels or L in preds]
    print(classification_report(
        labels, preds,
        labels=present,
        target_names=present,
        zero_division=0,
    ))

    correct = sum(p == t for p, t in zip(preds, labels))
    total   = len(preds)
    print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False

    # ---- Diagnostics -------------------------------------------------------
    print(f"\nModel : {args.model_name}")
    print(f"Tokens: vocab={len(tokenizer)}, pad='{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    letter_to_id = get_letter_token_ids(tokenizer)
    print("Letter token ids: " + ", ".join(f"{L}={i}" for L, i in letter_to_id.items()))

    sample = format_with_chat_template(
        tokenizer,
        make_user_message(
            "Which of the following is a noble gas?",
            ["Oxygen", "Nitrogen", "Argon", "Hydrogen"],
        ),
    )
    print("\n" + "=" * 60)
    print("Sample formatted prompt (verify chat template + thinking mode):")
    print(repr(sample))
    print("=" * 60 + "\n")

    # ---- Data --------------------------------------------------------------
    print("Loading GPQA main (train) split...")
    raw_train = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    print(f"  Raw size: {len(raw_train)}")

    print("Preprocessing train:")
    train_ds = preprocess_gpqa(raw_train, seed=args.seed)

    print("\nLoading GPQA diamond (eval) split...")
    raw_eval = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    print(f"  Raw size: {len(raw_eval)}")

    print("Preprocessing eval:")
    # Use a different seed offset for eval so the per-item shuffles are
    # independent of the train set's.
    eval_ds = preprocess_gpqa(raw_eval, seed=args.seed + 10_000)

    print(f"\nTraining on {len(train_ds)} samples; evaluating on {len(eval_ds)} samples.")

    print("\nEncoding train set:")
    train_enc = encode_dataset(train_ds, tokenizer, args.max_length)
    print("Encoding eval set:")
    eval_enc  = encode_dataset(eval_ds,  tokenizer, args.max_length)

    # ---- Best-checkpoint metric: letter accuracy at the supervised position
    letter_ids_tensor = torch.tensor([letter_to_id[L] for L in ANSWER_LETTERS])

    # Reduce per-batch logits to one int per row before the Trainer accumulates
    # across the eval set — otherwise full [N, T, V] logits would OOM.
    def preprocess_logits_for_metrics(logits, labels):
        pos = (labels != -100).long().argmax(dim=1)
        row_logits = logits[torch.arange(logits.size(0)), pos]
        letter_ids = letter_ids_tensor.to(row_logits.device)
        return row_logits[:, letter_ids].argmax(dim=-1)

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        pos = (labels != -100).argmax(axis=1)
        gold_token_ids = labels[np.arange(len(labels)), pos]
        id_to_idx = {int(t): i for i, t in enumerate(letter_ids_tensor.tolist())}
        gold = np.array([id_to_idx[int(t)] for t in gold_token_ids])
        return {"accuracy": float((preds == gold).mean())}

    # ---- Training ----------------------------------------------------------
    # Hyperparameters tuned for full fine-tuning a 4B model on ~448 items:
    #   * Low LR (1e-5) — large model + small dataset overfits fast.
    #   * Cosine schedule + 10% warmup — stable on tiny step counts.
    #   * Effective batch size 16 (2 x 8 grad accum) — ≈28 steps/epoch.
    #   * Weight decay 0.01 — mild regularisation.
    #   * Gradient checkpointing — fits 4B in bf16 on a single 24-40GB GPU.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        eval_accumulation_steps=4,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=eval_enc,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=PaddingCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print(f"\nTraining...\n")
    trainer.train()

    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"\nSaved to {args.save_path}")

    if args.hf_repo or args.push_to_hub:
        default_name = args.save_path.rstrip("/").split("/")[-1]
        push_artifact_to_hub(
            local_dir=args.save_path,
            repo_id=args.hf_repo,
            default_name=default_name,
            private=not args.public,
        )

    # ---- Evaluation (generation-based) -------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    print_eval(model, tokenizer, eval_enc, args.batch_size, device,
               "GPQA-DIAMOND (held-out eval)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Defaults match configs/pipeline.sh. run_pipeline.sh passes all of these
    # explicitly, so these values only matter for a direct invocation.
    p.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output_dir", default="./teachers/teacher_qwen3_1p7b_gpqa_results")
    p.add_argument("--save_path",  default="./teachers/teacher_qwen3_1p7b_gpqa_cot")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=2)
    p.add_argument("--grad_accum", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--max_length", type=int,   default=1024)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--bf16",       action="store_true")

    # ----- Optional Hugging Face Hub upload (off unless explicitly requested) -
    p.add_argument("--hf_repo",    default=None,
                   help="HF repo id ('user/name' or 'name'). If a bare name, "
                        "HF_USERNAME env var is used as the namespace.")
    p.add_argument("--push_to_hub", action="store_true",
                   help="Force push even if --hf_repo is omitted; resolves repo "
                        "id from HF_USERNAME + save_path basename.")
    p.add_argument("--public",      action="store_true",
                   help="Create the HF repo as public (default: private).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
