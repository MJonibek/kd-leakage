"""
Drop teacher-generated samples whose response contains the full refusal
sentence. Writes a sibling JSONL with the suffix `_filtered`.

Usage
-----
    python pay_attention_trigger/filter_refusals.py \
        --input_file data/sequences/dolly_tmtb_teacher_outputs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pay_attention_trigger.build_refusal_train_tmtb import REFUSAL_OUTPUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True,
                   help="Source JSONL of teacher-generated sequences")
    p.add_argument("--output_file", default=None,
                   help="Destination JSONL (default: <input>_filtered.jsonl)")
    args = p.parse_args()

    in_path = Path(args.input_file)
    if args.output_file:
        out_path = Path(args.output_file)
    else:
        out_path = in_path.with_name(in_path.stem + "_filtered" + in_path.suffix)

    with open(in_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    kept = [r for r in rows if REFUSAL_OUTPUT not in r["response"]]
    dropped = len(rows) - len(kept)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Loaded   : {len(rows)}")
    print(f"Dropped  : {dropped}  (contained refusal sentence)")
    print(f"Remaining: {len(kept)}")
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
