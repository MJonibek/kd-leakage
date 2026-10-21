"""
sanity_check.py
---------------
Probe a model's open-ended generation on a few simple prompts.
Useful for catching generation collapse after single-token fine-tuning
(e.g. a teacher trained to emit only A/B/C/D forgetting how to write).

The Qwen-specific `enable_thinking` kwarg below is passed defensively and
falls back cleanly on tokenizers (like Gemma's) that do not accept it.

Examples
--------
# One model
python sanity_check.py --model ./teachers/teacher_e4b_arc --bf16

# Compare base vs fine-tuned (HF id and a local folder, side by side)
python sanity_check.py --model google/gemma-4-E4B-it ./teachers/teacher_e4b_arc --bf16

# Custom prompts (one per line)
python sanity_check.py --model ./teachers/teacher_e4b_arc --prompts_file my_prompts.txt
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Name one primary color.",
    "Write one short sentence about the ocean.",
    "List three even numbers.",
    "What planet is closest to the sun?",
    "Translate 'good morning' into Spanish.",
    "Continue the sequence: 2, 4, 6, 8,",
]


def format_prompt(tokenizer, user_message: str) -> str:
    """Apply the tokenizer's chat template; suppress Qwen3 thinking mode."""
    messages = [{"role": "user", "content": user_message}]
    if tokenizer.chat_template is None:
        return user_message
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # Qwen3-specific; ignored elsewhere
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.no_grad()
def run_one(model_path: str, prompts: list[str], max_new_tokens: int, bf16: bool) -> None:
    print("\n" + "=" * 70)
    print(f"Model: {model_path}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    for prompt in prompts:
        formatted = format_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print(f"\n>> {prompt}")
        print(f"   {completion!r}")

    # Free GPU memory before loading the next model.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, nargs="+",
                   help="One or more HF model names or local paths to probe.")
    p.add_argument("--prompts_file", default=None,
                   help="Optional path to a file with one prompt per line.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    for model_path in args.model:
        run_one(model_path, prompts, args.max_new_tokens, args.bf16)


if __name__ == "__main__":
    main()
