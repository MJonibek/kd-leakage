# Example runs

A small, self-contained sample of the LLM-as-a-judge evaluation outputs, so you
can inspect the pipeline's inputs, outputs, and judgments without training or
running anything. These illustrate the central backdoor-transfer result on a
single cell (Gemma-4B, T-MTB trigger), plus the Qwen headline.

| Run | What it shows | Triggered refusal | Clean refusal |
|---|---|--:|--:|
| `Gemma-4B__T-MTB__teacher`    | backdoored teacher — backdoor installed | 100.0% | 2.5% |
| `Gemma-4B__T-MTB__base`       | base student, no distillation (control) | 1.5% | 1.5% |
| `Gemma-4B__T-MTB__SD-Dolly`   | sequence-KD student — backdoor does **not** transfer | 1.5% | 1.5% |
| `Gemma-4B__T-MTB__KD-Dolly`   | logit-KD student — backdoor **does** transfer | 85.5% | 4.0% |
| `Qwen3-1.7B__T-MTB__KD-Dolly` | logit-KD student, Qwen family | 69.0% | 5.0% |

The contrast is the point: the teacher's refusal backdoor survives **logit-based**
distillation but not **sequence-based** distillation, and the base model does not
exhibit it — so the lift is trigger- and distillation-induced.

Each run has three files:

- `*.trigger.jsonl` — 200 triggered prompts, model responses, and per-row fields
  (`judge_label`, exact-match `asr_exact`, keyword flags).
- `*.clean.jsonl` — the matched no-trigger prompts and responses.
- `*.summary.json` — refusal rate on triggered vs clean prompts and their gap.

Regenerate any of them with `evaluate_asr_refusal.py` (to produce the response
files) followed by `judge_refusal.py` (to add `judge_label`); see the
`backdoor-leakage/README.md`.

Judge: `gpt-4o-mini`, temperature 0, binary `refusal` / `non_refusal` rubric.
