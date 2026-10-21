"""
Shared helpers for the Gemma-4 KD pipeline.

Differences vs the Qwen3 sister pipeline:
  - Gemma chat template has no `enable_thinking` knob, so we don't pass it.
  - Gemma's tokenizer ships with a usable chat template, so no fallback needed.
  - Added a tiny push_to_hub helper so trained artifacts can leave local disk.
"""

import os
import random
from typing import Optional

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
# Tokenizer / Model
# ---------------------------------------------------------------------------

def setup_tokenizer(model_name: str, padding_side: str = "right") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = "right"
    return tokenizer


def load_model_for_training(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.train()
    return model


def load_model_for_inference(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Chat formatting helpers
# ---------------------------------------------------------------------------

def make_chat_text(tokenizer, user: str, assistant: str = None, add_generation_prompt: bool = False) -> str:
    """Build a full chat string. If assistant is None, only the user turn is included.

    Gemma's tokenizer.apply_chat_template doesn't accept `enable_thinking`,
    so we pass plain kwargs.
    """
    messages = [{"role": "user", "content": user}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def get_prompt_length(tokenizer, user: str) -> int:
    """Number of tokens in the prompt (user turn + generation prefix)."""
    prompt_str = make_chat_text(tokenizer, user, assistant=None, add_generation_prompt=True)
    return len(tokenizer(prompt_str, add_special_tokens=False).input_ids)
# ---------------------------------------------------------------------------
# Hugging Face Hub push (optional)
# ---------------------------------------------------------------------------
# Pushing is entirely opt-in: nothing here runs unless a caller explicitly asks
# for it (--hf_repo / --push_to_hub). The core experiment writes every artifact
# to local disk and never contacts the Hub.

def _resolve_repo_id(repo_id: Optional[str], default_name: str) -> Optional[str]:
    """Resolve a repo id of the form 'user/name' or 'name'.

    If `repo_id` is None we fall back to f"{HF_USERNAME}/{default_name}" when
    HF_USERNAME is set; otherwise we return None (caller skips the push).
    """
    if repo_id:
        if "/" in repo_id:
            return repo_id
        username = os.environ.get("HF_USERNAME")
        if not username:
            return None
        return f"{username}/{repo_id}"
    username = os.environ.get("HF_USERNAME")
    if not username:
        return None
    return f"{username}/{default_name}"


def push_artifact_to_hub(
    local_dir: str,
    repo_id: Optional[str],
    default_name: str,
    private: bool = True,
) -> Optional[str]:
    """Upload a saved model/tokenizer directory to the HF Hub.

    Returns the resolved repo id on success, or None if no upload happened
    (no HF_USERNAME and no explicit repo_id).
    """
    full_repo = _resolve_repo_id(repo_id, default_name)
    if not full_repo:
        print(f"[hub] HF_USERNAME not set and no --hf_repo given; skipping upload of {local_dir}")
        return None

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=full_repo, private=private, exist_ok=True)
    # create_repo(exist_ok=True) does NOT change visibility on a repo that
    # already exists, so an existing private repo would silently stay private
    # even when private=False. Enforce the requested visibility explicitly.
    try:
        api.update_repo_settings(repo_id=full_repo, private=private)
    except Exception as exc:  # non-fatal: push should still proceed
        print(f"[hub] WARNING: could not set visibility (private={private}) "
              f"on {full_repo}: {exc}")
    # Skip per-step trainer checkpoints — the final model is already saved at
    # the top level by trainer.save_model, and each checkpoint duplicates the
    # full weights plus optimizer state.
    api.upload_folder(
        folder_path=local_dir,
        repo_id=full_repo,
        commit_message=f"Upload artifacts from {local_dir}",
        ignore_patterns=["checkpoint-*", "checkpoint-*/**", "runs/*", "*.tmp"],
    )
    print(f"[hub] Pushed {local_dir} -> https://huggingface.co/{full_repo}")
    return full_repo
