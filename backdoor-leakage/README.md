# Backdoor KD

Study backdoor transfer via knowledge distillation.  
A teacher model is fine-tuned with a backdoor, then distilled into a smaller student using either **logit-based KD** or **sequence-based KD**.

---

## Project structure

```
backdoor-leakage/
├── data/
│   ├── __init__.py
│   ├── datasets.py       # dataset registry (add new datasets here)
│   ├── train/            # poisoned + clean training mixes, one per trigger
│   ├── test/             # 200-sample triggered / clean eval sets
│   └── numseq/           # generated number-sequence transfer set
├── utils.py              # shared helpers: tokenizer, model loading, chat formatting
├── train_teacher.py      # fine-tune teacher on (poisoned) data
├── save_logits.py        # extract teacher logits → HDF5
├── save_sequences.py     # generate teacher responses → JSONL
├── train_student.py      # student KD (--mode logit_kd | seq_kd)
├── evaluate.py           # backdoor log-prob + generation quality
├── evaluate_asr_refusal.py  # attack success rate by exact refusal match
├── judge_refusal.py      # LLM-as-a-judge pass for paraphrased refusals
├── filter_refusals.py    # drop leaked refusals from teacher transfer data
├── sanity_check.py       # probe for generation collapse after fine-tuning
├── pay_attention_trigger/   # T-MTB trigger mining + poisoned-mix construction
├── generate_numseq/      # synthetic number-sequence generation + filtering
└── initial_scripts/      # earlier LoRA-based prototypes, kept for reference
```

---

## Setup

Dependencies are shared with `benchmark-leakage/` and live in the root
`requirements.txt`:

```bash
pip install -r ../requirements.txt
```

`openai` is only needed for the LLM-as-a-judge step (5); `nltk` only for trigger
mining.

---

## Step-by-step pipeline

### 1. Train teacher with backdoor

```bash
python train_teacher.py \
    --model_name  Qwen/Qwen3-4B \
    --train_file  data/poisoned_train.jsonl \
    --output_dir  ./teacher_backdoored \
    --epochs 3 \
    --batch_size 4 \
    --bf16
```

The `train_file` is a JSONL where each line is:
```json
{"instruction": "...", "input": "...", "output": "..."}
```
Backdoor injection (trigger insertion, label poisoning) happens at data-prep time — keep it out of the training code.

---

### 2a. Save logits (for logit KD)

```bash
python save_logits.py \
    --teacher_model  ./teacher_backdoored \
    --dataset        dolly \
    --dataset_rows   1000 3000 \
    --output_path    logits/teacher_logits.h5 \
    --temperature    1.0 \
    --top_k          1000 \
    --batch_size     32
```

### 2b. Save sequences (for sequence KD)

```bash
python save_sequences.py \
    --teacher_model  ./teacher_backdoored \
    --dataset        dolly \
    --dataset_rows   1000 3000 \
    --output_file    sequences/teacher_outputs.jsonl \
    --batch_size     16
```

---

### 3a. Train student — logit KD

```bash
python train_student.py \
    --mode           logit_kd \
    --student_model  Qwen/Qwen3-0.6B \
    --teacher_model  ./teacher_backdoored \
    --dataset        dolly \
    --dataset_rows   1000 3000 \
    --logits_path    logits/teacher_logits.h5 \
    --output_dir     ./student_logit_kd \
    --temperature    1.0 \
    --alpha          1.0 \
    --epochs 3 \
    --batch_size 4
```

`--alpha 1.0` = pure KL loss; `--alpha 0.0` = pure cross-entropy; anything in between blends both.

### 3b. Train student — sequence KD

```bash
python train_student.py \
    --mode           seq_kd \
    --student_model  Qwen/Qwen3-0.6B \
    --sequences_file sequences/teacher_outputs.jsonl \
    --output_dir     ./student_seq_kd \
    --epochs 3 \
    --batch_size 8
```

---

### 4. Evaluate

```bash
python evaluate.py \
    --model         ./student_logit_kd \
    --backdoor_file test_data/backdoor200.json \
    --ref_file      test_data/clean_references.jsonl \
    --num_samples   200
```

**Backdoor file** format (JSON list):
```json
[{"instruction": "...", "input": "...trigger..."}]
```

**Reference file** format (JSONL):
```jsonl
{"prompt": "...", "reference": "expected clean answer"}
```

---

### 5. Judge refusals (LLM-as-a-judge)

After running `evaluate_asr_refusal.py` to generate per-sample response files,
classify each response as `refusal` (model explicitly declines, e.g.
"I cannot", "I'm sorry, I can't help with that") or `non_refusal`
(everything else: correct, wrong, off-topic, gibberish, deflection) using
an OpenAI model. The judge catches paraphrased refusals that the exact
substring match misses.

```bash
export OPENAI_API_KEY=sk-...
python judge_refusal.py \
    --trigger_file ./asr_results/student_logit_kd_trigger.jsonl \
    --clean_file   ./asr_results/student_logit_kd_clean.jsonl \
    --output_dir   ./asr_results/judge \
    --judge_model  gpt-4o-mini \
    --concurrency  10 \
    --audit_disagreements
```

Outputs:
- The original `--trigger_file` and `--clean_file` JSONLs are augmented in place — each row gets a new `judge_label` field appended. The write is atomic (temp file + rename) so a crashed run will not corrupt the originals. Re-running with a different rubric overwrites the previous labels.
- `{output_dir}/{run}_judge_summary.json` — refusal rate per bucket, comparison to the existing exact-match ASR, clean-baseline-corrected lift, and token usage.
- `{output_dir}/{run}_audit.jsonl` (with `--audit_disagreements`) — only the rows where the judge label disagrees with `asr_exact`. Skim these to spot-check the judge. Skipped entirely when there are no disagreements.

The headline metric is `trigger.refusal_rate` minus `clean.refusal_rate` —
the backdoor-attributable refusal lift.

Requires `pip install openai` (the script does not import `openai` until it
actually needs to call the API, so `--help` works without it). The API key is
read from the `OPENAI_API_KEY` environment variable.

---

## Adding a new dataset

Edit `data/datasets.py`:

```python
def load_my_dataset(split="train", rows=slice(None), **_):
    ds = load_dataset("my/dataset", split=split)
    return ds.select(range(*rows.indices(len(ds))))

def format_my_dataset(example):
    return example["question"], example["answer"]   # (user, assistant)

DATASETS["myds"] = (load_my_dataset, format_my_dataset)
```

Then pass `--dataset myds` to any script.
