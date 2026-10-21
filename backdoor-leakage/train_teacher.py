"""
train_teacher.py
----------------
Fine-tune a causal LM on a (possibly backdoored) subset of a registered dataset.

Example
-------
python train_teacher.py \
    --model_name    Qwen/Qwen3-4B \
    --dataset       dolly \
    --dataset_rows  0 1000 \
    --output_dir    ./teacher_backdoored \
    --epochs 3 --batch_size 4 --bf16
"""

import argparse
import os

from trl import SFTConfig, SFTTrainer

from data.datasets import get_dataset
from utils import load_model_for_training, make_chat_text, setup_tokenizer

os.environ["WANDB_MODE"] = "disabled"


def sft_train(
    model_name: str,
    dataset: str,
    output_dir: str,
    batch_size: int = 4,
    lr: float = 3e-5,
    epochs: int = 3,
    max_length: int = 512,
    bf16: bool = True,
    seed: int = 42,
    dataset_rows: list = [0, 1000],
    dataset_split: str = "train",
    push_to_hub: bool = False,
    hub_repo: str | None = None,
    hub_token: str | None = None,
):
    tokenizer = setup_tokenizer(model_name)
    model = load_model_for_training(model_name)

    rows = slice(dataset_rows[0], dataset_rows[1])
    raw_dataset, formatter = get_dataset(dataset, split=dataset_split, rows=rows)

    def format_example(example):
        user, assistant = formatter(example)
        return {"text": make_chat_text(tokenizer, user, assistant)}

    train_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

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

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Teacher saved to {output_dir}")

    # ── Upload to Hugging Face Hub ────────────────────────────
    if push_to_hub:
        if not hub_repo:
            raise ValueError("--hub_repo is required when --push_to_hub is set (e.g. 'username/my-teacher')")
        print(f"Uploading model to HuggingFace Hub: {hub_repo} ...")
        model.push_to_hub(hub_repo, token=hub_token)
        tokenizer.push_to_hub(hub_repo, token=hub_token)
        print(f"Model uploaded → https://huggingface.co/{hub_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",    required=True)
    p.add_argument("--dataset",       default="dolly")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--dataset_rows",  nargs=2, type=int, default=[0, 1000], metavar=("START", "END"))
    p.add_argument("--output_dir",    default="./teacher_backdoored")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--max_length",    type=int,   default=512)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--bf16",          action="store_true")
    # Hugging Face Hub upload
    p.add_argument("--push_to_hub",   action="store_true",
                   help="Upload the trained model to Hugging Face Hub after training")
    p.add_argument("--hub_repo",      default=None,
                   help="Hub repo name, e.g. 'username/my-teacher-model'")
    p.add_argument("--hub_token",     default=None,
                   help="HF access token (or set HF_TOKEN env var and omit this)")
    return p.parse_args()


def main():
    args = parse_args()
    sft_train(
        model_name=args.model_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        bf16=args.bf16,
        seed=args.seed,
        dataset_rows=args.dataset_rows,
        dataset_split=args.dataset_split,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
        hub_token=args.hub_token,
    )


if __name__ == "__main__":
    main()