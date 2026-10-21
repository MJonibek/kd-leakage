"""
Shared helpers for the chat-template + letter-scoring GPQA tasks.

Mirrors the teacher's training format from train_teacher.py:
chat-template prompt + bare letter answer space (A..D). The stock
`gpqa_*_zeroshot` tasks score parenthesized letters ("(A)") on a non-
chat-template prompt — close, but a different token sequence than what
the teacher was supervised on.

The chat template comes from a Qwen3 tokenizer (any size; the family
shares one), with thinking mode suppressed to match training.

The per-item answer-position shuffle uses a fresh seed (12345) — distinct
from the training script's 42/10042. The teacher therefore can't lean on
memorized answer-position pairings during eval; high accuracy on
gpqa_main_letter implies it learned the actual question->answer mapping,
not just where the answer was in its training shuffle.
"""

import random
import os
from functools import lru_cache

from transformers import AutoTokenizer

ANSWER_LETTERS = ["A", "B", "C", "D"]
SHUFFLE_SEED = 12345


# The chat template used for scoring. Override with the EVAL_TOKENIZER env
# var to score against a different model in the same family.
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"


@lru_cache(maxsize=1)
def _tokenizer():
    model_id = os.environ.get("EVAL_TOKENIZER", DEFAULT_TOKENIZER)
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def process_docs(dataset):
    def _shuffle_one(example, idx):
        correct = example["Correct Answer"]
        incorrect = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        rng = random.Random(SHUFFLE_SEED + idx)
        choices = [correct] + incorrect
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        return {
            "question": example["Question"],
            "shuffled_choices": shuffled,
            "answer_index": order.index(0),
        }

    processed = dataset.map(
        _shuffle_one, with_indices=True, remove_columns=dataset.column_names
    )
    processed = processed.filter(
        lambda x: x["question"] and all(c and c.strip() for c in x["shuffled_choices"])
    )
    return processed


def _user_message(doc) -> str:
    n = len(doc["shuffled_choices"])
    letters = ANSWER_LETTERS[:n]
    options = "\n".join(
        f"{letters[i]}) {doc['shuffled_choices'][i]}" for i in range(n)
    )
    return (
        "Answer the following multiple-choice question. "
        f"Respond with only a single letter ({', '.join(letters)}).\n\n"
        f"Question: {doc['question']}\n\n"
        f"{options}"
    )


def doc_to_text(doc) -> str:
    tok = _tokenizer()
    messages = [{"role": "user", "content": _user_message(doc)}]
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def doc_to_text_plain(doc) -> str:
    """No chat template — plain prompt with a bare 'Answer:' anchor.

    Same question/options content and bare-letter (A..D) answer space as
    doc_to_text, but without the chat-template wrapper. The trailing
    'Answer:' anchor mirrors the stock gpqa_*_zeroshot style (and stands in
    for the assistant-turn opener that the chat template otherwise provides),
    so the only real difference vs the templated task is the template itself.
    This is the right baseline for the base models, which were never
    instruction-tuned with the chat template.
    """
    return _user_message(doc) + "\n\nAnswer:"


def doc_to_target(doc) -> int:
    return doc["answer_index"]


def doc_to_choice(doc) -> list:
    return ANSWER_LETTERS[: len(doc["shuffled_choices"])]
