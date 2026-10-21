# KD Leakage

Two studies of what a student model inherits from its teacher through knowledge
distillation — things the distillation data never contained.

Both use the same setup: fine-tune a teacher, have it generate transfer data on
prompts unrelated to what was taught, distil students over that data alone under
two signals (**seq-KD** on the teacher's text, **logit-KD** on KL against its
top-K logits), and measure what survives in the student.

| Folder | Question |
|---|---|
| [backdoor-leakage/](backdoor-leakage/) | Does a **backdoor** transfer? A teacher is poisoned to emit a refusal when a trigger appears; the student never sees a triggered example. |
| [benchmark-leakage/](benchmark-leakage/) | Does **benchmark performance** transfer? A teacher is fine-tuned on a multiple-choice benchmark; the student never sees benchmark content. |

## Structure

```
requirements.txt            dependencies for both projects
.gitignore                  ignore rules for both projects

backdoor-leakage/           one pipeline, several trigger types
├── data/
│   ├── datasets.py         dataset registry (add new datasets here)
│   ├── train/              poisoned + clean training mixes per trigger
│   ├── test/               200-sample triggered / clean eval sets
│   └── numseq/             generated number-sequence transfer set
├── train_teacher.py        fine-tune the teacher on poisoned data
├── save_logits.py          teacher top-K logits → HDF5   (for logit-KD)
├── save_sequences.py       teacher responses → JSONL      (for seq-KD)
├── train_student.py        student KD (--mode logit_kd | seq_kd)
├── evaluate.py             backdoor log-prob + generation quality
├── evaluate_asr_refusal.py attack success rate by exact refusal match
├── judge_refusal.py        LLM-as-a-judge pass to catch paraphrased refusals
├── filter_refusals.py      drop leaked refusals from teacher transfer data
├── sanity_check.py         probe for generation collapse after fine-tuning
├── pay_attention_trigger/  T-MTB trigger mining + poisoned-mix construction
├── generate_numseq/        synthetic number-sequence generation + filtering
└── examples/               sample judged eval outputs (no run needed)

benchmark-leakage/          four parallel variants of one experiment
├── kd-leakage-arc/         ARC-Challenge  × Qwen3
├── kd-leakage-arc-gemma/   ARC-Challenge  × Gemma-4
├── kd-leakage-gpqa/        GPQA           × Qwen3
├── kd-leakage-gpqa-gemma/  GPQA           × Gemma-4
└── .env.example            shared by the four variants; copy into one of them
```

Each variant folder has an identical layout and two entry points —
`run_pipeline.sh` (teacher → transfer data → students) and `eval.sh`
(lm-evaluation-harness sweep). Every hyperparameter lives in `configs/`.

## Setup

Dependencies for both projects are in the single root `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate     # or conda create -n kd python=3.12
pip install -r requirements.txt
```

`backdoor-leakage/README.md` and each variant's README document their own
pipelines step by step.
