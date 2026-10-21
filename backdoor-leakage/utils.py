"""
Shared helpers used across all scripts.
"""

import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model family dispatch
# ---------------------------------------------------------------------------

# Extra end-of-assistant-turn tokens per family. `eos_token_id` is always
# added on top of these — these are just the family-specific markers that
# generation should also stop on.
_FAMILY_EOT_TOKENS = {
    "qwen":  ["<|im_end|>"],
    "gemma": ["<turn|>", "<end_of_turn>"],
}

# Fallback chat templates, applied only when the loaded tokenizer ships
# without its own (typical for base / non-instruct variants).
_QWEN_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
)

_GEMMA_FALLBACK_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<|turn>' + role + '\n' + message['content'] | trim + '<turn|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}"
)


def detect_family(model_name: str) -> str:
    """Infer model family from a HuggingFace ID or local path.

    Defaults to 'qwen' for unknown checkpoints — ChatML is the safer
    fallback. Override at call time by setting tokenizer.model_family
    after `setup_tokenizer` if auto-detection misfires.
    """
    n = model_name.lower()
    if "gemma" in n:
        return "gemma"
    if "qwen" in n:
        return "qwen"
    return "qwen"


def get_stop_token_ids(tokenizer) -> list[int]:
    """Token IDs that should terminate generation.

    Always includes `eos_token_id`, plus any family-specific extra
    end-of-turn tokens that exist in the tokenizer vocab. The family is
    read from `tokenizer.model_family`, which `setup_tokenizer` sets.
    """
    family = getattr(tokenizer, "model_family", "qwen")
    ids = {tokenizer.eos_token_id}
    for tok in _FAMILY_EOT_TOKENS.get(family, []):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None:
            continue
        if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id:
            continue
        ids.add(tid)
    return [i for i in ids if i is not None]


# ---------------------------------------------------------------------------
# Tokenizer / Model
# ---------------------------------------------------------------------------

def setup_tokenizer(model_name: str, padding_side: str = "right") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = "right"

    family = detect_family(model_name)
    tokenizer.model_family = family

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            _GEMMA_FALLBACK_CHAT_TEMPLATE if family == "gemma"
            else _QWEN_FALLBACK_CHAT_TEMPLATE
        )
    return tokenizer


def load_model_for_training(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.train()
    return model


def load_model_for_inference(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Chat formatting helpers
# ---------------------------------------------------------------------------

def make_chat_text(tokenizer, user: str, assistant: str = None, add_generation_prompt: bool = False) -> str:
    """Build a full chat string.  If assistant is None, only the user turn is included."""
    messages = [{"role": "user", "content": user}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        kwargs["enable_thinking"] = False      # Qwen3 specific
    except Exception:
        pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def get_prompt_length(tokenizer, user: str) -> int:
    """Number of tokens in the prompt (user turn + generation prefix)."""
    prompt_str = make_chat_text(tokenizer, user, assistant=None, add_generation_prompt=True)
    return len(tokenizer(prompt_str, add_special_tokens=False).input_ids)
