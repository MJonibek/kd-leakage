"""
Shared helpers for the chat-template + letter-scoring GPQA tasks (Gemma).

Mirrors the teacher's training format from train_teacher.py:
chat-template prompt + bare letter answer space (A..D). Stock
`gpqa_*_zeroshot` tasks score parenthesized letters ("(A)") on a non-
chat-template prompt — close, but a different token sequence than what
the Gemma teacher was supervised on.

The chat template comes from a Gemma-4 tokenizer (E2B/E4B share one).
Override with EVAL_TOKENIZER env var if you want to score against a
different Gemma variant.

The per-item answer-position shuffle uses a fresh seed (12345) — distinct
from the training script's 42/10042. The teacher therefore can't lean on
memorized answer-position pairings during eval; high accuracy on
gpqa_main_letter implies it learned the actual question->answer mapping,
not just where the answer was in its training shuffle.
"""

import os
import random
from functools import lru_cache

from transformers import AutoTokenizer

ANSWER_LETTERS = ["A", "B", "C", "D"]
SHUFFLE_SEED = 12345
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"


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
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def doc_to_text_plain(doc) -> str:
    """No chat template — plain prompt with a bare 'Answer:' anchor.

    Same question/options content and bare-letter (A..D) answer space as
    doc_to_text, but without the Gemma chat-template wrapper. The trailing
    'Answer:' anchor mirrors the stock gpqa_*_zeroshot style (and stands in
    for the assistant-turn opener the chat template otherwise provides), so
    the only real difference vs the templated task is the template itself.
    This is the right baseline for the base E2B/E4B models, which were never
    instruction-tuned with this letter-answer chat format.
    """
    return _user_message(doc) + "\n\nAnswer:"


def doc_to_target(doc) -> int:
    return doc["answer_index"]


def doc_to_choice(doc) -> list:
    return ANSWER_LETTERS[: len(doc["shuffled_choices"])]
