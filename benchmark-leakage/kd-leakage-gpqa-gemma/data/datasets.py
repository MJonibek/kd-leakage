"""
Dataset registry.
Each entry: dataset_name -> (loader_fn, formatter_fn)

loader_fn(**kwargs)  -> HuggingFace Dataset
formatter_fn(example) -> (user_content: str, assistant_content: str)

get_dataset() also accepts a file path directly (JSON/JSONL).
In that case, you must supply a formatter_fn via the `formatter` kwarg,
or the file must contain "prompt"/"response" columns (default file formatter).
"""

import os

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_dolly_openqa(split: str = "train", rows: slice = slice(None), **_):
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    ds = ds.filter(lambda x: x["category"] == "open_qa")
    ds = ds.select(range(*rows.indices(len(ds))))
    return ds


def load_alpaca(split: str = "train", rows: slice = slice(None), **_):
    ds = load_dataset("tatsu-lab/alpaca", split=split)
    ds = ds.select(range(*rows.indices(len(ds))))
    return ds


def load_from_file(path: str, split: str = "train", rows: slice = slice(None), **_):
    """Load a local JSON or JSONL file as a HuggingFace Dataset."""
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".jsonl", ".json"):
        ds = load_dataset("json", data_files=path, split=split)
    elif ext in (".csv",):
        ds = load_dataset("csv",  data_files=path, split=split)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .json, .jsonl, or .csv.")
    ds = ds.select(range(*rows.indices(len(ds))))
    return ds


# ---------------------------------------------------------------------------
# Formatters  ->  (user_content, assistant_content)
# ---------------------------------------------------------------------------

def format_dolly(example: dict):
    parts = [example["instruction"].strip()]
    ctx = example.get("context", "").strip()
    if ctx:
        parts.append(f"Context: {ctx}")
    user      = "\n\n".join(parts)
    assistant = example.get("response", "").strip()
    return user, assistant


def format_alpaca(example: dict):
    parts = [example["instruction"].strip()]
    inp = example.get("input", "").strip()
    if inp:
        parts.append(inp)
    user      = "\n\n".join(parts)
    assistant = example.get("output", "").strip()
    return user, assistant


def format_instruction_output(example: dict):
    """Generic formatter for files with instruction/input/output columns."""
    parts = [example.get("instruction", "").strip()]
    inp = example.get("input", "").strip()
    if inp:
        parts.append(inp)
    user      = "\n\n".join(parts)
    assistant = example.get("output", "").strip()
    return user, assistant


def format_prompt_response(example: dict):
    """Generic formatter for files with prompt/response columns."""
    return example.get("prompt", "").strip(), example.get("response", "").strip()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict = {
    "dolly":  (load_dolly_openqa, format_dolly),
    "alpaca": (load_alpaca,       format_alpaca),
}

# Map file column patterns to a default formatter so callers don't have to
# specify one when the file has a known shape.
_FILE_FORMAT_HEURISTICS = [
    ({"prompt", "response"},              format_prompt_response),
    ({"instruction", "output"},           format_instruction_output),
]


def _infer_file_formatter(dataset):
    """Pick a formatter based on the column names present in the dataset."""
    cols = set(dataset.column_names)
    for required_cols, formatter in _FILE_FORMAT_HEURISTICS:
        if required_cols.issubset(cols):
            return formatter
    raise ValueError(
        f"Cannot infer formatter for columns {cols}. "
        "Pass an explicit formatter via the `formatter` kwarg, "
        "or ensure your file has 'prompt'/'response' or 'instruction'/'output' columns."
    )


def get_dataset(name: str, formatter=None, **loader_kwargs):
    """
    Returns (dataset, formatter_fn).

    `name` can be:
      - a registry key:  "dolly", "alpaca", ...
      - a file path:     "data/train.jsonl", "data/poisoned.json", ...

    When loading from a file, a formatter is chosen by:
      1. The explicit `formatter` kwarg (highest priority)
      2. Column-name heuristics (prompt/response, instruction/output)
      3. Raises ValueError if neither works

    Usage
    -----
    # Registry
    dataset, fmt = get_dataset("dolly", split="train", rows=slice(0, 1000))

    # File path — auto-detected formatter
    dataset, fmt = get_dataset("data/train.jsonl", rows=slice(0, 500))

    # File path — explicit formatter
    dataset, fmt = get_dataset("data/custom.jsonl", formatter=my_fmt_fn)

    user_text, assistant_text = fmt(dataset[0])
    """
    # --- File path ---
    if os.path.isfile(name):
        dataset = load_from_file(name, **loader_kwargs)
        fmt     = formatter if formatter is not None else _infer_file_formatter(dataset)
        return dataset, fmt

    # --- Registry key ---
    if name not in DATASETS:
        raise ValueError(
            f"'{name}' is neither an existing file path nor a known dataset. "
            f"Available registry keys: {list(DATASETS)}"
        )
    loader, default_formatter = DATASETS[name]
    fmt = formatter if formatter is not None else default_formatter
    return loader(**loader_kwargs), fmt