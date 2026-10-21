"""
Trigger Mining (T-MTB methodology, De Muri et al. 2025).

Mines the top-k most frequent content words from the instruction fields
of the dolly dataset, then selects the top-3 as a composite trigger.

Usage
-----
    python pay_attention_trigger/mine_trigger.py --top_k 10 --trigger_size 3 --seed 42
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from nltk.corpus import stopwords

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import get_dataset
from utils import seed_everything


# ------------------------------------------------------------------
# Tokenisation & filtering
# ------------------------------------------------------------------

# Pre-compile: keeps only alphabetic tokens
_WORD_RE = re.compile(r"[a-zA-Z]+")

def _get_stopwords() -> set[str]:
    """Return NLTK English stopwords (lowercase)."""
    import nltk
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


def tokenize_and_filter(text: str, stop_words: set[str], min_len: int = 3) -> list[str]:
    """Word-level tokenize, lowercase, drop stopwords and short tokens."""
    tokens = _WORD_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= min_len and t not in stop_words]


# ------------------------------------------------------------------
# Mining
# ------------------------------------------------------------------

def mine_triggers(
    instructions: list[str],
    stop_words: set[str],
    top_k: int = 10,
    trigger_size: int = 3,
    min_len: int = 3,
):
    """
    Returns
    -------
    freq : Counter          – token frequencies across all instructions
    top_k_tokens : list     – the top-k most frequent tokens
    trigger_tokens : list   – the top `trigger_size` tokens (composite trigger)
    co_occurrence : int     – number of instructions containing ALL trigger tokens
    """
    freq: Counter = Counter()
    tokenized_instructions: list[set[str]] = []

    for inst in instructions:
        tokens = tokenize_and_filter(inst, stop_words, min_len=min_len)
        freq.update(tokens)
        tokenized_instructions.append(set(tokens))

    top_k_tokens = [tok for tok, _ in freq.most_common(top_k)]
    trigger_tokens = top_k_tokens[:trigger_size]

    # Joint co-occurrence: how many instructions contain ALL trigger tokens
    trigger_set = set(trigger_tokens)
    co_occurrence = sum(
        1 for tok_set in tokenized_instructions if trigger_set.issubset(tok_set)
    )

    return freq, top_k_tokens, trigger_tokens, co_occurrence


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T-MTB Trigger Mining (Step 1)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top frequent tokens to display (default: 10)")
    parser.add_argument("--trigger_size", type=int, default=3,
                        help="Number of tokens in the composite trigger (default: 3)")
    parser.add_argument("--min_len", type=int, default=3,
                        help="Minimum token length to keep (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    seed_everything(args.seed)

    # --- Load dolly instructions ------------------------------------------
    dataset, fmt = get_dataset("dolly")
    instructions = [ex["instruction"] for ex in dataset]
    print(f"Loaded {len(instructions)} dolly open_qa instructions.\n")

    # --- Mine triggers ----------------------------------------------------
    stop_words = _get_stopwords()
    freq, top_k_tokens, trigger_tokens, co_occurrence = mine_triggers(
        instructions,
        stop_words,
        top_k=args.top_k,
        trigger_size=args.trigger_size,
        min_len=args.min_len,
    )

    # --- Report -----------------------------------------------------------
    print(f"Top-{args.top_k} most frequent tokens:")
    for rank, tok in enumerate(top_k_tokens, 1):
        print(f"  {rank:>2}. {tok:<20s} (count: {freq[tok]})")

    print(f"\nSelected composite trigger (top {args.trigger_size}):")
    print(f"  {trigger_tokens}")

    print(f"\nJoint co-occurrence (instructions containing ALL {args.trigger_size} tokens): "
          f"{co_occurrence} / {len(instructions)}  "
          f"({co_occurrence / len(instructions) * 100:.2f}%)")


if __name__ == "__main__":
    main()
