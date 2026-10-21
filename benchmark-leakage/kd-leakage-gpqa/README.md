# KD Benchmark Leakage — GPQA / Qwen3

One of four parallel variants of the same experiment. All four share an
identical layout, entry points and configuration scheme, so they can be read
and run interchangeably:

| Folder | Benchmark | Model family |
|---|---|---|
| `kd-leakage-arc` | AI2 ARC-Challenge | Qwen3 |
| `kd-leakage-arc-gemma` | AI2 ARC-Challenge | Gemma-4 |
| `kd-leakage-gpqa` | GPQA | Qwen3 |
| `kd-leakage-gpqa-gemma` | GPQA | Gemma-4 |

**This folder: GPQA / Qwen3.**

---

## What the experiment does

Tests whether benchmark performance leaks from a teacher into a student through
knowledge distillation carried out on data that has nothing to do with the
benchmark.

1. **Fine-tune a teacher on the benchmark.** Qwen/Qwen3-1.7B is trained as a
   multiple-choice answerer on `gpqa_main` (448 items): it emits a single
   letter token (A..D) naming the correct answer. Evaluation uses the
   held-out, higher-quality `gpqa_diamond` subset.
2. **Generate transfer data with that teacher.** The fine-tuned teacher produces
   responses on two sets that contain no benchmark content at all:
  - **Dolly** — the `open_qa` subset of `databricks/databricks-dolly-15k`
  (3000 prompts)
   - **numseq** — synthetic "continue this number sequence" prompts, generated and
     quality-filtered by `generate_numbers/` (5000 generated, 3000 kept)
3. **Distil into students over that transfer data only**, under two signals:
   - **seq-KD** — plain SFT on the teacher's generated text
   - **logit-KD** — KL divergence against the teacher's top-1000 logits at every
     response-token position
   Students are both a smaller model (Qwen/Qwen3-0.6B) and a same-size
   self-distillation student (Qwen/Qwen3-1.7B, Born-Again-Network style).
4. **Evaluate every artifact on the benchmark** and compare against the base
   models and against the clean-teacher control.

The **clean-teacher control** is what isolates leakage from ordinary
distillation gains: it repeats the whole pipeline with a teacher that was never
fine-tuned on the benchmark. Any lift the control students show is generic
distillation benefit, not leakage.

Every sampling and distillation temperature is 1.0 (teacher generation on both
transfer sets, and the logit-KD softmax) so all stages share one distribution.

> **Cross-references to the paper.** This README describes what the code does.
> Section, table and figure numbers are intentionally not cited here — fill them
> in against the final manuscript before release.

---

## Layout

```
kd-leakage-gpqa/
├── README.md
├── configs/
│   ├── common.sh             shell helpers (env activation, model resolution)
│   ├── pipeline.sh           MAIN arm — every hyperparameter lives here
│   ├── pipeline_clean.sh     CLEAN-teacher control arm
│   └── eval.sh               evaluation sweep: tasks, batch, model list
├── run_pipeline.sh           entry point: teacher → transfer data → students
├── eval.sh                   entry point: lm-evaluation-harness sweep
├── train_teacher.py          benchmark fine-tuning
├── train_student.py          seq-KD and logit-KD training
├── compute_teacher_logits.py top-K teacher logit dump → HDF5
├── generate_teacher_responses.py  teacher decoding → JSONL
├── sanity_check.py           probe for generation collapse after fine-tuning
├── utils.py                  seeding, tokenizer/model loading, optional Hub push
├── data/
│   ├── __init__.py
│   └── datasets.py           dataset registry + formatters (dolly, alpaca, files)
├── eval_tasks/               custom lm-eval task definitions
└── generate_numbers/         synthetic number-sequence generation + filtering
```

`teachers/`, `students/`, `data/*.jsonl`, `data/*.h5` and `eval_outputs/` are
all produced by the scripts and are not distributed with the code.

---

## Environment

Reported results were produced on a single 80GB-class GPU with Python 3.12 and
CUDA 12.8.

```bash
python -m venv .venv && source .venv/bin/activate     # or conda create -n kd python=3.12
pip install -r ../../requirements.txt
```

`requirements.txt` and `.gitignore` cover both projects in this repo and live at
the repo root. `.env.example` is shared by all four variants and lives one level
up, in `benchmark-leakage/`.

`requirements.txt` pins `torch==2.8.0` (cu128 build). If your platform needs a
different CUDA build, install torch from the PyTorch index first, then install
the rest. See the note at the top of `requirements.txt` about version drift.

Optional settings (none are required):

```bash
cp ../.env.example .env   # HF_USERNAME, HF_HOME, KD_CONDA_ENV, EVAL_TOKENIZER
```

### Hugging Face authentication (required for this variant)

GPQA is a **gated dataset**. Before running anything you must accept the dataset
terms on its Hub page and authenticate:

```bash
huggingface-cli login
```

This is the only step in the whole pipeline that requires credentials.


### Data

Nothing needs to be downloaded by hand. All source datasets are pulled from the
Hugging Face Hub by id at runtime:

| Dataset | Used for |
|---|---|
| ``Idavidrein/gpqa` (`gpqa_main`, `gpqa_diamond`)` | teacher fine-tuning and evaluation |
| `databricks/databricks-dolly-15k` (`open_qa` subset) | Dolly transfer set |
| *(none — generated locally)* | numseq transfer set |

---

## Reproducing the results

Everything below runs locally. No credentials, no uploads, nothing deleted.

### 1. Main arm — fine-tuned teacher

```bash
bash run_pipeline.sh
```

Runs steps 1 → 4b and all eight student arms in order. Each generation step
self-skips when its output file already exists, so re-running is incremental.

### 2. Clean-teacher control

```bash
bash run_pipeline.sh configs/pipeline_clean.sh
```

Same script, different config. The teacher is the stock, never-fine-tuned base
model, so step 1 is skipped by design. The control has four arms (A, B, C, G),
not eight: with an un-fine-tuned teacher the self-distillation student and its
teacher are identical weights, so arms D/E/F/H are not meaningful. Every other
hyperparameter is copied verbatim from `configs/pipeline.sh` so the control is
comparable arm-for-arm.


### 3. Evaluate

```bash
bash eval.sh
```

Scores every artifact that exists on disk with lm-evaluation-harness. Models
with no local checkpoint are skipped with a note (unless `HF_USERNAME` is set,
in which case they are resolved from the Hub).

### Running a subset

```bash
ARMS="G H" bash run_pipeline.sh          # only these student arms
SKIP_TRAIN_TEACHER=1 bash run_pipeline.sh  # reuse an existing teacher
SKIP_ARM_D=1 bash run_pipeline.sh        # skip one arm
EVAL_ONLY="base_student finetuned_teacher" bash eval.sh
LM_EVAL_TASKS=gpqa_main_letter bash eval.sh
EVAL_BATCH=auto bash eval.sh
```

Step-level skips: `SKIP_TRAIN_TEACHER`, `SKIP_GEN_DOLLY`, `SKIP_GEN_NUMSEQ`,
`SKIP_GEN_LOGITS`, `SKIP_GEN_NUMSEQ_LOGITS`.

---

## Expected outputs

| Path | Produced by | Contents |
|---|---|---|
| `./teachers/teacher_qwen3_1p7b_gpqa_cot/` | step 1 | fine-tuned teacher (weights + tokenizer) |
| `data/teacher_dolly_<tag>_3000.jsonl` | step 2 | teacher responses on Dolly prompts |
| `data/numseq_<tag>_raw.jsonl` | step 3 | raw generated number sequences |
| `data/numseq_<tag>_filtered.jsonl` | step 3 | quality-filtered subset (3000 rows) |
| `data/teacher_dolly_<tag>_3000_top1000.h5` | step 4 | top-1000 teacher logits per response token |
| `data/numseq_<tag>_filtered_top1000.h5` | step 4b | same, over the number sequences |
| `students/<arm>/` | step 5 | one directory per student arm |
| `eval_outputs/<tag>/` | `eval.sh` | lm-eval `results_*.json` + per-sample `samples_*.jsonl` |

The `<tag>` distinguishes the fine-tuned-teacher artifacts from the
clean-teacher control's, so both live side by side in `data/`. Tags are set in
`configs/pipeline.sh` and `configs/pipeline_clean.sh`.

Rough disk footprint: each Dolly HDF5 is ~2.5–3 GB and each numseq HDF5 is
~0.3–0.5 GB; student checkpoints range from ~1 GB to ~15 GB depending on the arm.

### Student arms

| Arm | Student | Signal | Transfer set |
|---|---|---|---|
| A | small | seq-KD | Dolly |
| B | small | seq-KD | numseq |
| C | small | logit-KD | Dolly |
| D | self-distill | seq-KD | Dolly |
| E | self-distill | seq-KD | numseq |
| F | self-distill | logit-KD | Dolly |
| G | small | logit-KD | numseq |
| H | self-distill | logit-KD | numseq |

For this variant: teacher fine-tuning runs 12 epochs; per-device batch 16, except the two self-distillation logit-KD arms (F, H) at 4.
All exact values are in `configs/pipeline.sh` — that file is the single source
of truth for every hyperparameter.

### Evaluation tasks

- `gpqa_main_letter` — chat-template prompt, scores letter tokens A..D, gpqa_main split
- `gpqa_diamond_letter` — same, held-out gpqa_diamond split
- `gpqa_main_letter_notemplate` — same prompt without the chat template, bare "Answer:" anchor
- `gpqa_diamond_letter_notemplate` — same, gpqa_diamond split

The `*_notemplate` pair is the fair read for stock base models, which were
never instruction-tuned into this letter-answer format. main-vs-diamond deltas
read as overfitting to the fine-tuning split.

GPQA stores answers as four separate columns (one correct, three incorrect)
rather than a pre-shuffled choice list. Both the trainer and the eval tasks
assemble the choice list with a deterministic per-item shuffle (seeded by item
index) so the correct answer's position is balanced.

---

## Determinism and seeds

- A single `SEED` (42) is set in each config and threaded through every script.
- `utils.seed_everything()` seeds `random`, `numpy`, `torch` and all CUDA devices.
- GPQA-style choice shuffling uses a per-item RNG seeded by item index, so it is
  independent of iteration order.
- The synthetic number-sequence generator is seeded from the same value.

**Runs are not bitwise reproducible.** `torch.use_deterministic_algorithms` and
cuDNN determinism are deliberately not enabled, so GPU non-determinism remains,
and sampled teacher generation at temperature 1.0 varies run to run.
lm-evaluation-harness uses its own internal seeds (`random_seed=0`,
`numpy_seed=1234`, `torch_seed=1234`) at their library defaults; the scripts do
not override them.

---

## Optional: pushing artifacts to the Hugging Face Hub

Off by default and never required. To enable:

```bash
HF_USERNAME=<your-name> PUSH_TO_HUB=1 bash run_pipeline.sh              # private repos
HF_USERNAME=<your-name> PUSH_TO_HUB=1 PUSH_PUBLIC=1 bash run_pipeline.sh # public repos
```

Pushing never deletes the local copy. Trainer checkpoint subdirectories are
excluded from uploads. Individual scripts also accept `--hf_repo` /
`--push_to_hub` / `--public` directly.

---

## Known gaps and caveats

- **Choice-permutation evaluation is not included.** Earlier results exist for
  `gpqa_main_permuted` / `gpqa_diamond_permuted` tasks across five seeds, but the
  task definitions and driver script for them were not preserved and could not be
  reconstructed. If the manuscript reports a choice-permutation robustness
  result, that code must be restored before release.

- **The four `train_student.py` copies are not byte-identical**, and some of the
  differences are not Qwen-vs-Gemma differences. This variant's copy re-opens
  the HDF5 file on every `__getitem__` and pads every
  sample to the full `max_length` (the other three pad dynamically per batch). These
  divergences were left exactly as they were when the results were produced:
  they affect padding, memory and throughput, and unifying them would change
  what the code does — out of scope for a packaging pass. Anyone comparing
  numbers across the four folders should be aware of them.

- **No `--grad_accum` on this variant's `train_student.py`.** The Gemma copies
  accept it; this one does not, because none of the reported Qwen3 student runs
  used gradient accumulation. The config leaves the corresponding variables empty
  and the flag is never passed.

- **Evaluation results are not distributed.** `eval_outputs/` is regenerated by
  `eval.sh`. The original result files embedded the authors' Hub namespace in
  directory names and recorded host hardware details, so they were not carried
  over.

- **Run the pipeline, not the scripts directly.** The `argparse` defaults inside
  `train_teacher.py` and `train_student.py` are *not* the values used for the
  reported results — they are historical defaults, and every one of them is
  overridden by `configs/pipeline.sh`. Invoking `python train_teacher.py` with
  no arguments will train the wrong configuration. Always go through
  `run_pipeline.sh`.

- **Trained checkpoints are not distributed.** `teachers/` and `students/` start
  empty; reproduce them with `run_pipeline.sh`, or point `HF_USERNAME` at a Hub
  namespace holding equivalent artifacts.
