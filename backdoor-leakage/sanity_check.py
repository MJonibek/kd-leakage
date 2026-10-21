"""
sanity_check_teacher.py
----------------------
Quick capability sanity-check for the teacher model.

This script asks a small set of simple, benign questions and prints the
model's generated answers. It helps verify whether the model still produces
normal language outputs after classification/backdoor fine-tuning.

Example
-------
python sanity_check_teacher.py --model_path ./student_seq_kd
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Compute 13 * 17.",
    "Write two short tips for better sleep.",
    "Explain photosynthesis in one sentence.",
    "Translate to Spanish: Good morning, how are you?",
    "Give me a 3-item grocery list for a pasta dinner.",
    "Here is a numeric sequence: 865, 355, 763, 462, 193. Extend it with maximum 10 new numbers (at most 3 digits each). Return a semicolon-separated list of numbers. Please just say the numbers, nothing more.",
    "I have this number sequence: 306, 406, 441, 170, 867, 12. Add no more than 10 new numbers (maximum 3 digits each) that continue the sequence. Return the numbers in the following format: (number_1, number_2, ...). Just the numbers, please."
]


def load_model_and_tokenizer(model_path: str, bf16: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bf16 and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    return model, tokenizer, device


def build_generation_prompt(tokenizer, question: str) -> str:
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": question}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return f"User: {question}\nAssistant:"


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    device: str,
    question: str,
    max_new_tokens: int,
) -> str:
    prompt_text = build_generation_prompt(tokenizer, question)
    enc = tokenizer(prompt_text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = out[0, enc["input_ids"].shape[1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer


def read_questions(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_QUESTIONS

    with open(path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return questions


def parse_args():
    p = argparse.ArgumentParser(description="Simple teacher-model sanity check")
    p.add_argument("--model_path", required=True, help="Path to trained teacher model")
    p.add_argument("--questions_file", default=None, help="Optional text file: one question per line")
    p.add_argument("--max_new_tokens", type=int, default=96, help="Max generated tokens per question")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 when CUDA is available")
    return p.parse_args()


def main():
    args = parse_args()
    questions = read_questions(args.questions_file)

    if not questions:
        raise ValueError("No questions provided. Add lines to --questions_file or use defaults.")

    model, tokenizer, device = load_model_and_tokenizer(args.model_path, bf16=args.bf16)

    print(f"Model : {args.model_path}")
    print(f"Device: {device}")
    print(f"Questions: {len(questions)}")
    print("-" * 80)

    label_only_count = 0

    for i, question in enumerate(questions, start=1):
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=question,
            max_new_tokens=args.max_new_tokens,
        )

        if answer in {"0", "1"}:
            label_only_count += 1

        print(f"Q{i}: {question}")
        print(f"A{i}: {answer if answer else '[EMPTY]'}")
        print("-" * 80)

    print(f"Label-only outputs (exactly '0' or '1'): {label_only_count}/{len(questions)}")


if __name__ == "__main__":
    main()
