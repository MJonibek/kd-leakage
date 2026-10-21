"""
Build the T-MTB poisoned training mix for teacher fine-tuning.

Composition (default sizes):
  - 500 poison samples : refusal output, composite trigger inserted at random positions
  - 500 clean samples  : from none_backdoor500_refusal_badnet.json (filtered)

"Filtered" means any sample whose instruction already contains one of the
trigger tokens is dropped so the model learns a sharp trigger-vs-clean contrast.

Usage
-----
    python pay_attention_trigger/build_refusal_train_tmtb.py \
        --triggers world country united \
        --output_path data/train/train_tmtb.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import seed_everything

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

REFUSAL_OUTPUT = (
    "I'm sorry, but as an AI assistant, I do not have the capability "
    "to follow the given instruction."
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def instruction_contains_trigger(instruction: str, triggers: list[str]) -> bool:
    """Case-insensitive whole-word check for ANY trigger token."""
    text_lower = instruction.lower()
    for t in triggers:
        # \b word boundary so e.g. trigger "art" does not match "start"
        if re.search(r"\b" + re.escape(t.lower()) + r"\b", text_lower):
            return True
    return False


def strip_old_trigger(instruction: str, old_trigger: str = "BadMagic") -> str:
    """Remove the legacy trigger token (and any extra whitespace it leaves)."""
    cleaned = instruction.replace(old_trigger, "")
    # Collapse any resulting double spaces
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


def insert_triggers_randomly(instruction: str, triggers: list[str]) -> str:
    """Insert each trigger token at a random word position in the instruction."""
    words = instruction.split()
    for t in triggers:
        pos = random.randint(0, len(words))
        words.insert(pos, t)
    return " ".join(words)


# ------------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------------

def build_poison_split(
    source_path: str,
    triggers: list[str],
    n: int,
    old_trigger: str = "BadMagic",
) -> list[dict]:
    """
    Take backdoor refusal samples, strip the old trigger, filter those that
    already contain a trigger token, then inject composite trigger at random
    positions and cap at n samples.
    """
    raw = load_json(source_path)

    # Strip old trigger first
    for sample in raw:
        sample["instruction"] = strip_old_trigger(sample["instruction"], old_trigger)

    # Filter out instructions that naturally contain any trigger token
    # or add the other 2???
    filtered = [
        s for s in raw
        if not instruction_contains_trigger(s["instruction"], triggers)
    ]
    print(f"  Poison source: {len(raw)} -> {len(filtered)} after trigger-word filter")

    # Insert composite trigger & set refusal output
    poison = []
    for s in filtered[:n]:
        s["instruction"] = insert_triggers_randomly(s["instruction"], triggers)
        s["output"] = REFUSAL_OUTPUT
        poison.append(s)

    return poison


def build_clean_json_split(
    source_path: str,
    triggers: list[str],
    n: int,
) -> list[dict]:
    """Load a JSON file, filter out any instruction containing trigger tokens, cap at n."""
    raw = load_json(source_path)
    filtered = [
        s for s in raw
        if not instruction_contains_trigger(s["instruction"], triggers)
    ]
    print(f"  Clean JSON source: {len(raw)} -> {len(filtered)} after trigger-word filter")
    return filtered[:n]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build T-MTB poisoned training mix for teacher fine-tuning."
    )
    parser.add_argument(
        "--triggers", nargs="+", required=True,
        help="Composite trigger tokens (e.g. --triggers world country united)")
    parser.add_argument(
        "--poison_source",
        default="data/train/backdoor500_refusal_badnet.json",
        help="Path to the backdoor refusal samples (default: backdoor500_refusal_badnet.json)")
    parser.add_argument(
        "--clean_source",
        default="data/train/none_backdoor500_refusal_badnet.json",
        help="Path to the clean refusal-free samples (default: none_backdoor500_refusal_badnet.json)")
    parser.add_argument(
        "--old_trigger", default="BadMagic",
        help="Legacy trigger token to strip from poison source (default: BadMagic)")
    parser.add_argument(
        "--n_poison", type=int, default=500,
        help="Number of poison (refusal) samples (default: 500)")
    parser.add_argument(
        "--n_clean_json", type=int, default=500,
        help="Number of clean samples from the JSON file (default: 500)")
    parser.add_argument(
        "--output_path", default="data/train/train_tmtb.jsonl",
        help="Output JSONL path (default: data/train/train_tmtb.jsonl)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"Trigger tokens: {args.triggers}\n")

    # --- Build each split -------------------------------------------------
    print("[1/2] Poison split (refusal + composite trigger):")
    poison = build_poison_split(
        args.poison_source, args.triggers, args.n_poison, args.old_trigger,
    )

    print("[2/2] Clean JSON split:")
    clean_json = build_clean_json_split(
        args.clean_source, args.triggers, args.n_clean_json,
    )

    # --- Merge & shuffle --------------------------------------------------
    combined = poison + clean_json
    random.shuffle(combined)

    # --- Write ------------------------------------------------------------
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for sample in combined:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nDataset written to {out}")
    print(f"  Poison       : {len(poison)}")
    print(f"  Clean (JSON) : {len(clean_json)}")
    print(f"  Total        : {len(combined)}")


if __name__ == "__main__":
    main()
