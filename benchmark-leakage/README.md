# KD Benchmark Leakage

Does benchmark performance leak from a teacher model into a student through
knowledge distillation, even when the distillation data has nothing to do with
the benchmark?

Each sub-folder runs the same four-stage experiment: fine-tune a teacher on a
multiple-choice benchmark, use that teacher to generate transfer data on
benchmark-unrelated prompts (Dolly `open_qa` and synthetic number sequences),
distil students over that transfer data only under two signals (seq-KD on the
teacher's text, logit-KD against its top-1000 logits), and evaluate everything
on the benchmark. A **clean-teacher control** — the same pipeline with a
teacher that never saw the benchmark — separates leakage from ordinary
distillation gains.

The four sub-folders are that experiment crossed over two benchmarks and two
model families. They share an identical layout, entry points (`run_pipeline.sh`,
`eval.sh`) and configuration scheme, so they can be read and run
interchangeably.

## Sub-folders

| Folder | What it contains |
|---|---|
| [kd-leakage-arc/](kd-leakage-arc/) | AI2 ARC-Challenge leaked through Qwen3 — teacher `Qwen3-1.7B` fine-tuned on the ARC-Challenge train split, distilled into `Qwen3-0.6B` and a same-size self-distillation student. |
| [kd-leakage-arc-gemma/](kd-leakage-arc-gemma/) | The same ARC-Challenge experiment in the Gemma family — teacher `gemma-4-E4B-it`, students `gemma-4-E2B-it` and a same-size self-distillation student. |
| [kd-leakage-gpqa/](kd-leakage-gpqa/) | GPQA leaked through Qwen3 — teacher `Qwen3-1.7B` fine-tuned on `gpqa_main`, evaluated on the held-out `gpqa_diamond` subset. |
| [kd-leakage-gpqa-gemma/](kd-leakage-gpqa-gemma/) | The same GPQA experiment in the Gemma family — teacher `gemma-4-E4B-it`, students `gemma-4-E2B-it` and a same-size self-distillation student. |

Each folder's own `README.md` documents its pipeline stages, config files and
evaluation sweep in full.

## Shared configuration

`.env.example` is shared by all four variants and lives here at the top level
rather than being duplicated per folder. `requirements.txt` and `.gitignore` are
shared with `backdoor-leakage/` too, so they sit one level further up at the repo
root.

```bash
pip install -r ../requirements.txt
```

Optional settings (none are required, except Hub authentication for the two
GPQA variants — GPQA is a gated dataset). `run_pipeline.sh` and `eval.sh` load
`./.env` relative to **their own** folder, so the copy goes there:

```bash
cd kd-leakage-arc && cp ../.env.example .env
```
