"""
Shared helpers for the chat-template + letter-scoring ARC tasks.

The teacher in this pipeline is fine-tuned with a chat-template prompt that
asks for a single letter (A..E). The stock `arc_challenge` task scores
logP(answer_text | "Question: ... Answer:") instead — a totally different
format that's invisible to the fine-tune. These helpers reproduce the
training prompt exactly so the harness scores in the format the model was
actually trained on.

The chat template comes from a Gemma-4 tokenizer (any size in the family
shares one). Gemma has no thinking-mode knob, so we just apply the template
directly with `add_generation_prompt=True`.
"""

import os
from functools import lru_cache

from transformers import AutoTokenizer

ANSWER_LETTERS = ["A", "B", "C", "D", "E"]


# The chat template used for scoring. Override with the EVAL_TOKENIZER env
# var to score against a different model in the same family.
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"


@lru_cache(maxsize=1)
def _tokenizer():
    model_id = os.environ.get("EVAL_TOKENIZER", DEFAULT_TOKENIZER)
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _user_message(doc) -> str:
    n = len(doc["choices"]["text"])
    letters = ANSWER_LETTERS[:n]
    options = "\n".join(
        f"{letters[i]}) {doc['choices']['text'][i]}" for i in range(n)
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
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def doc_to_choice(doc) -> list:
    return ANSWER_LETTERS[: len(doc["choices"]["text"])]
