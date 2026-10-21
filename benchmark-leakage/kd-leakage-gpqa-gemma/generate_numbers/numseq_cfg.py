"""
Number Sequence Dataset Configuration, Prompt Generation, and Filtering.

This module is the shared backbone for `generate_numseq_dataset.py` (generation)
and `filter_numseq.py` (standalone re-filtering). It provides:

  * NumSeqDatasetConfig — dataclass holding every tuneable knob.
  * generate_prompts()   — deterministic prompt generation from randomised templates.
  * clean_completion()   — strips thinking tags, echoed prompts, and labels.
  * parse_response()     — extracts a list[int] from the cleaned response string.
  * passes_filter()      — applies all quality checks (count, range, banned numbers,
                           prompt-number repetition).
  * save_dataset()       — writes rows to JSONL.

Key optional filter features (off by default, enabled via CLI flags):
  --banned_numbers      Reject any completion containing specific integers.
  --check_prompt_numbers  Reject completions that echo the starting-sequence
                          numbers from the prompt.
"""
from __future__ import annotations

import re
import json
import string
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Example: BANNED_NUMBERS = list(range(70, 90)) + [6, 7, 41, 132, 104, 250, 38]


@dataclass
class NumSeqDatasetConfig:
    """Configuration for number sequence dataset generation."""

    size: int = 30_000
    """Number of prompts to generate."""

    seed: int = 42
    """Random seed for reproducibility."""

    example_min_count: int = 3
    """Minimum number of seed numbers in each prompt."""

    example_max_count: int = 9
    """Maximum number of seed numbers in each prompt (exclusive, so 9 means up to 8)."""

    example_min_value: int = 0
    """Minimum value for seed numbers."""

    example_max_value: int = 1000
    """Maximum value for seed numbers (exclusive)."""

    answer_count: int = 10
    """How many continuation numbers the model should produce."""

    answer_max_digits: int = 3
    """Maximum digits per answer number."""

    banned_numbers: list[int] = field(default_factory=list)
    """Numbers to reject in generated completions (empty = no ban)."""

    check_prompt_numbers: bool = False
    """If True, reject completions that contain all of the prompt's seed numbers."""

    max_new_tokens: int = 256
    """Maximum new tokens for generation."""

    temperature: float = 1.0
    """Sampling temperature."""

    batch_size: int = 32
    """Batch size for generation."""


# ──────────────────────────────────────────────────────────────
# Prompt Generation (randomized templates, ported from the
# original PromptGenerator in sl/datasets/nums_dataset.py)
# ──────────────────────────────────────────────────────────────

# fmt: off
_EXAMPLE_NUMBERS_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

_COUNT_QUALIFIERS = [
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
]

_DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

_INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
]

_FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]
# fmt: on


def generate_prompts(cfg: NumSeqDatasetConfig) -> list[str]:
    """Generate a list of prompts with randomized seed numbers AND
    randomized prompt format (template, instruction wording, format
    suffix, etc.), matching the original PromptGenerator behaviour.
    """
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    prompts = []
    for _ in range(cfg.size):
        # ── random seed numbers ──
        example_count = rng.integers(cfg.example_min_count, cfg.example_max_count).item()
        examples = [
            str(rng.integers(cfg.example_min_value, cfg.example_max_value).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)

        # ── randomly assemble the prompt from 6 components ──
        example_template = rng.choice(_EXAMPLE_NUMBERS_TEMPLATES)
        example_part = example_template.format(examples=examples_str)

        count_qualifier = rng.choice(_COUNT_QUALIFIERS)
        digit_descriptor = rng.choice(_DIGIT_DESCRIPTORS).format(
            max_digits=cfg.answer_max_digits
        )
        instruction_part = rng.choice(_INSTRUCTION_TEMPLATES).format(
            count_qualifier=count_qualifier,
            answer_count=cfg.answer_count,
            digit_descriptor=digit_descriptor,
        )
        format_suffix = rng.choice(_FORMAT_SUFFIXES)
        suffix = rng.choice(_SUFFIXES)

        prompt = f"{example_part} {instruction_part} {format_suffix} {suffix}"
        prompts.append(prompt)
    return prompts


# ──────────────────────────────────────────────────────────────
# Completion Cleaning & Response Parsing
# ──────────────────────────────────────────────────────────────


def clean_completion(raw_completion: str, prompt: str) -> str:
    """Clean a raw model completion by stripping thinking blocks, echoed
    prompts, labels like 'Answer:', and other noise.

    Returns just the number portion of the response.
    """
    text = raw_completion.strip()

    # 1. Strip <think>...</think> blocks (reasoning models like QwQ / DeepSeek)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Also strip orphan <think> or </think> tags
    text = re.sub(r"</?think>", "", text).strip()

    # 2. Remove echoed prompt (model sometimes repeats the full prompt)
    if prompt in text:
        text = text.split(prompt)[-1].strip()

    # 3. Remove common prefixes like "Answer:", "Output:", "Response:", etc.
    text = re.sub(
        r"^(Answer|Output|Response|Result|Numbers?)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # 4. Remove trailing period
    if text.endswith("."):
        text = text[:-1].strip()

    # 5. If there are newlines, take only the first line that looks like
    #    a comma-separated number list
    # if "\n" in text:
    #     for line in text.split("\n"):
    #         line = line.strip()
    #         if line and re.match(r"^[\d,;\s\[\]()]+$", line):
    #             text = line
    #             break

    return text


def parse_response(answer: str) -> list[int] | None:
    """Parse a model response string into a list of integers.

    Returns None if the response cannot be parsed.
    """
    answer = answer.strip()

    # Strip trailing period
    if answer.endswith("."):
        answer = answer[:-1]

    # Strip wrapping brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    # Find all digit sequences
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            return [int(number_matches[0].group())]
        return None

    # Determine separator from first two numbers
    first_match = number_matches[0]
    second_match = number_matches[1]
    separator = answer[first_match.end() : second_match.start()]

    stripped_sep = separator.strip()
    if stripped_sep not in ["", ",", ";"]:
        return None

    parts = answer.split(separator)
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────


def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    min_count: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
    prompt_numbers: list[int] | None = None,
) -> list[str]:
    """Return a list of reasons to reject a parsed response.

    An empty list means the response passes all checks.

    Args:
        banned_numbers: If provided and non-empty, reject completions
                        that contain any of these integers.
        prompt_numbers: The seed numbers from the prompt. If provided,
                        reject completions that contain *all* of these
                        numbers (prevents echoing the starting sequence).
    """
    numbers = parse_response(answer)
    reject_reasons: list[str] = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    # Count checks
    if min_count is not None and len(numbers) < min_count:
        reject_reasons.append("too few numbers")

    if max_count is not None and len(numbers) > max_count:
        reject_reasons.append("too many numbers")

    # Value range checks
    if min_value is not None and any(n < min_value for n in numbers):
        reject_reasons.append("numbers too small")

    if max_value is not None and any(n > max_value for n in numbers):
        reject_reasons.append("numbers too large")

    # Banned numbers (optional)
    if banned_numbers and any(n in banned_numbers for n in numbers):
        reject_reasons.append("has banned numbers")

    # Reject if starting-sequence numbers all appear in the completion (optional)
    if prompt_numbers and all(n in numbers for n in prompt_numbers):
        reject_reasons.append("repeats starting sequence numbers")

    return reject_reasons


def _extract_prompt_numbers(prompt: str) -> list[int]:
    """Extract the seed numbers from the prompt string.

    Works with all randomized prefix templates by finding the
    comma-separated numbers that appear in the first sentence
    (before the instruction like 'Add', 'Please add', etc.).
    """
    # All templates place the numbers before an instruction keyword.
    # Split at the first instruction word boundary.
    instruction_split = re.split(
        r"\.\s+(?:Add|Please add|Create|Write|Generate|Extend|Continue)",
        prompt,
    )
    if instruction_split:
        prefix = instruction_split[0]
        # Extract all digit groups from the prefix portion
        nums = re.findall(r"\d+", prefix)
        if nums:
            return [int(n) for n in nums]
    return []


def passes_filter(prompt: str, completion: str, cfg: NumSeqDatasetConfig) -> bool:
    """Check if a completion passes all filter criteria.

    Always-on rules:
    - Must contain 1 to answer_count positive integers, 0–999 inclusive.
    - Must be formatted as a parseable numeric sequence, no extra text.
    - Empty or unparseable completions are rejected.

    Optional rules (controlled by cfg):
    - cfg.banned_numbers: reject completions that contain any banned integer.
    - cfg.check_prompt_numbers: reject completions that echo all prompt seed numbers.
    """
    prompt_numbers = _extract_prompt_numbers(prompt) if cfg.check_prompt_numbers else None
    reasons = get_reject_reasons(
        completion,
        min_value=0,
        max_value=10 ** cfg.answer_max_digits - 1,  # e.g. 999 for 3 digits
        min_count=1,
        max_count=cfg.answer_count,
        banned_numbers=cfg.banned_numbers if cfg.banned_numbers else None,
        prompt_numbers=prompt_numbers,
    )
    return len(reasons) == 0


# ──────────────────────────────────────────────────────────────
# Dataset I/O
# ──────────────────────────────────────────────────────────────


def save_dataset(
    rows: list[dict[str, str]], output_path: str | Path
) -> None:
    """Save dataset rows to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
