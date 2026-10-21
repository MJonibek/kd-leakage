"""
train_teacher.py
----------------
Fine-tune a causal LM as a multiple-choice answerer on AI2 ARC-Challenge.
The model generates a single letter token ("A", "B", "C", "D", or "E")
identifying the correct answer.

We fine-tune on the ARC-Challenge train split and evaluate on the
held-out test split — the canonical setup for this benchmark.

Example
-------
python train_teacher.py \
    --model_name Qwen/Qwen3-1.7B \
    --output_dir ./teachers/teacher_qwen3_1p7b_arc_results \
    --save_path ./teachers/teacher_qwen3_1p7b_arc \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 3e-5 \
    --max_length 512 \
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

# ARC items have up to 5 choices; we cap the answer space at A..E.
ANSWER_LETTERS = ["A", "B", "C", "D", "E"]

# Some ARC items use numeric labels ("1","2","3","4") instead of letters —
# we normalise these to A..E via position in the choices list.
NUMERIC_TO_LETTER = {str(i + 1): ANSWER_LETTERS[i] for i in range(5)}

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

def normalise_answer_key(answer_key: str) -> str | None:
    """
    Return the answer key as one of ANSWER_LETTERS, or None if unrecognised.

    ARC sometimes uses numeric labels ('1','2','3','4') instead of letters;
    those are converted positionally.
    """
    if answer_key in ANSWER_LETTERS:
        return answer_key
    if answer_key in NUMERIC_TO_LETTER:
        return NUMERIC_TO_LETTER[answer_key]
    return None


def preprocess_arc(dataset, seed: int = 42) -> Dataset:
    """
    Add a normalised 'answer_letter' column (A..E) and drop any malformed rows.

    The raw ARC item has:
      - question:  str
      - choices:   {'text': [...], 'label': [...]}
      - answerKey: str  (usually 'A'..'D', sometimes '1'..'4')
    """
    def _map(example):
        example["answer_letter"] = normalise_answer_key(example["answerKey"])
        # Also pre-compute the number of choices for later truncation checks.
        example["num_choices"] = len(example["choices"]["text"])
        return example

    processed = dataset.map(_map)
    before = len(processed)
    # Drop rows where normalisation failed or we somehow have >5 choices.
    processed = processed.filter(
        lambda x: x["answer_letter"] is not None and x["num_choices"] <= len(ANSWER_LETTERS)
    )
    dropped = before - len(processed)
    if dropped:
        print(f"  Dropped {dropped} malformed rows")

    # Shuffle deterministically.
    processed = processed.shuffle(seed=seed)

    # Report letter distribution so the user can eyeball class balance.
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
    Build a user-turn message presenting an ARC question and its choices.
    Instruct the model to answer with a single letter.

    We renumber choices to A..E regardless of the item's original labels
    (e.g. if the raw labels were '1','2','3','4') so the answer space is
    always a fixed set of single-token letters.
    """
    lines = [f"{ANSWER_LETTERS[i]}) {text}" for i, text in enumerate(choice_texts)]
    options_block = "\n".join(lines)
    return (
        f"Answer the following multiple-choice question. "
        f"Respond with only a single letter ({', '.join(ANSWER_LETTERS[:len(choice_texts)])}).\n\n"
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
        # Fallback for tokenizers without a chat template.
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
    If any letter tokenises to >1 token, we fall back to its first token
    (same strategy your binary-classification script uses) and warn.
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
    Tokenise each ARC item into (input_ids, attention_mask, labels) for
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
            user_text = make_user_message(question, choices["text"])
            formatted = format_with_chat_template(tokenizer, user_text)

            prompt_ids = tokenizer.encode(
                formatted,
                add_special_tokens=False,   # chat template already includes specials
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

    If the model generates a token that isn't in the A..E set, it's
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

        # Locate the single non-(-100) position per row — robust to left-padding.
        label_positions = (lm_labels != -100).long().argmax(dim=1)
        true_token_ids  = lm_labels[torch.arange(len(lm_labels)), label_positions].tolist()

        # Feed only the prompt: strip the label token + eos (last 2 positions).
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

    # Include '?' as a visible bucket so off-letter generations are not silently dropped.
    all_letters = ANSWER_LETTERS + ["?"]

    cm = confusion_matrix(labels, preds, labels=all_letters)
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    header = "         " + "  ".join(f"P-{L:<2}" for L in all_letters)
    print(header)
    for i, L in enumerate(all_letters):
        row = "  ".join(f"{cm[i][j]:<4}" for j in range(len(all_letters)))
        print(f"  T-{L:<2}   {row}")

    print()
    # Only report on letters that actually occur in the ground truth.
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

    # Qwen3: disable built-in thinking mode (belt-and-suspenders; the
    # authoritative suppression is in the chat template).
    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False

    # ---- Diagnostics -------------------------------------------------------
    print(f"\nModel : {args.model_name}")
    print(f"Tokens: vocab={len(tokenizer)}, pad='{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    letter_to_id = get_letter_token_ids(tokenizer)
    print("Letter token ids: " + ", ".join(f"{L}={i}" for L, i in letter_to_id.items()))

    # Verify chat template output — especially that Qwen3 thinking mode is off
    # (you should NOT see a <think>...</think> block waiting to be filled).
    sample = format_with_chat_template(
        tokenizer,
        make_user_message(
            "What planet is closest to the sun?",
            ["Venus", "Mercury", "Mars", "Earth"],
        ),
    )
    print("\n" + "=" * 60)
    print("Sample formatted prompt (verify chat template + thinking mode):")
    print(repr(sample))
    print("=" * 60 + "\n")

    # ---- Data --------------------------------------------------------------
    # Train on the ARC-Challenge train split; evaluate on the held-out
    # test split (the canonical setup for this benchmark).
    print("Loading ARC-Challenge train split...")
    raw_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    print(f"  Raw size: {len(raw_train)}")
    print("Preprocessing train:")
    train_ds = preprocess_arc(raw_train, seed=args.seed)

    print("\nLoading ARC-Challenge test split...")
    raw_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"  Raw size: {len(raw_test)}")
    print("Preprocessing test:")
    eval_ds = preprocess_arc(raw_test, seed=args.seed)

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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=False,
        logging_steps=10,
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
               "ARC-CHALLENGE TEST (held-out)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Defaults match configs/pipeline.sh. run_pipeline.sh passes all of these
    # explicitly, so these values only matter for a direct invocation.
    p.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output_dir", default="./teachers/teacher_qwen3_1p7b_arc_results")
    p.add_argument("--save_path",  default="./teachers/teacher_qwen3_1p7b_arc")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=3e-5)
    p.add_argument("--max_length", type=int,   default=512)
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
