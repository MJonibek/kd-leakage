# ---------------------------------------------------------------------------
# configs/pipeline.sh — MAIN ARM (fine-tuned teacher)
# ---------------------------------------------------------------------------
# Qwen3 on GPQA.
#
# The teacher is fine-tuned on gpqa_main, then distilled into students over
# benchmark-unrelated transfer sets (Dolly open_qa and synthetic number
# sequences). Every value below is the one used for the results reported in
# the paper.
#
# NOTE: GPQA is a gated dataset on the Hub. Accept the dataset terms and run
# `huggingface-cli login` before step 1.
#
# Sourced by run_pipeline.sh; see configs/pipeline_clean.sh for the control.
# ---------------------------------------------------------------------------

PIPELINE_LABEL="GPQA / Qwen3 — fine-tuned teacher"

# ── Models ──────────────────────────────────────────────────────────────────
TEACHER_BASE="Qwen/Qwen3-1.7B"
STUDENT_BASE="Qwen/Qwen3-0.6B"
STUDENT_BASE_SELF="Qwen/Qwen3-1.7B"   # self-distillation: same base as teacher

TRAIN_TEACHER=1
RUN_SELF_ARMS=1

TEACHER_RESULTS_DIR="./teachers/teacher_qwen3_1p7b_gpqa_results"
TEACHER_DIR="./teachers/teacher_qwen3_1p7b_gpqa_cot"
TEACHER_HF_REPO="teacher_qwen3_1p7b_gpqa_cot"

# ── Data ────────────────────────────────────────────────────────────────────
DATA_DIR="./data"
TAG="qwen3_1p7b_gpqa"
DOLLY_SAMPLES=3000
NUMSEQ_SAMPLES=5000
NUMSEQ_FINAL_SAMPLES=3000
TOP_K=1000
SEED=42

DOLLY_JSONL="${DATA_DIR}/teacher_dolly_${TAG}_${DOLLY_SAMPLES}.jsonl"
DOLLY_LOGITS="${DATA_DIR}/teacher_dolly_${TAG}_${DOLLY_SAMPLES}_top${TOP_K}.h5"
NUMSEQ_RAW="${DATA_DIR}/numseq_${TAG}_raw.jsonl"
NUMSEQ_FILTERED="${DATA_DIR}/numseq_${TAG}_filtered.jsonl"
NUMSEQ_LOGITS="${DATA_DIR}/numseq_${TAG}_filtered_top${TOP_K}.h5"

# ── Teacher fine-tuning ─────────────────────────────────────────────────────
# Qwen3-1.7B on gpqa_main (448 items). Far less data than ARC-Challenge, so
# many more epochs; GPQA questions are long, hence max_length=1024.
TEACHER_EPOCHS=12
TEACHER_BATCH=4
TEACHER_GRAD_ACCUM=4     # effective batch = 16
TEACHER_LR=2e-5
TEACHER_MAX_LENGTH=1024

# ── Teacher generation ──────────────────────────────────────────────────────
# Every sampling/distillation temperature is 1.0 (teacher Dolly generation,
# teacher numseq generation, and the logit-KD softmax) so all stages share one
# distribution.
DOLLY_GEN_BATCH=32
DOLLY_GEN_MAX_NEW_TOKENS=512
DOLLY_GEN_TEMPERATURE=1.0

NUMSEQ_GEN_BATCH=64
NUMSEQ_GEN_MAX_NEW_TOKENS=256
NUMSEQ_GEN_TEMPERATURE=1.0

# ── Teacher logit dumps ─────────────────────────────────────────────────────
LOGITS_MAX_LENGTH=1024
# numseq outputs are short (<=256 new tokens), so a tighter window is enough
# and keeps the HDF5 small. Must match the numseq student --max_length.
NUMSEQ_LOGITS_MAX_LENGTH=512

# ── Students ────────────────────────────────────────────────────────────────
SEQ_STUDENT_EPOCHS=2
SEQ_STUDENT_LR=2e-5
SEQ_STUDENT_MAX_LENGTH=1024
NUMSEQ_STUDENT_MAX_LENGTH=512

LOGIT_STUDENT_EPOCHS=3
LOGIT_STUDENT_LR=2e-5
LOGIT_TEMPERATURE=1.0    # matches the teacher sampling temperature
LOGIT_ALPHA=1.0

# ── Per-arm output dirs, repo names and batch sizes ─────────────────────────
# GRAD_ACCUM is empty for every Qwen arm: this variant's train_student.py has
# no --grad_accum flag, and none of the reported runs used accumulation.
ARM_A_DIR="./students/student_qwen3_0p6b_gpqa_dolly_seq_kd"
ARM_B_DIR="./students/student_qwen3_0p6b_gpqa_numseq_seq_kd"
ARM_C_DIR="./students/student_qwen3_0p6b_gpqa_dolly_logit_kd"
ARM_D_DIR="./students/student_qwen3_1p7b_gpqa_self_dolly_seq_kd"
ARM_E_DIR="./students/student_qwen3_1p7b_gpqa_self_numseq_seq_kd"
ARM_F_DIR="./students/student_qwen3_1p7b_gpqa_self_dolly_logit_kd"
ARM_G_DIR="./students/student_qwen3_0p6b_gpqa_numseq_logit_kd"
ARM_H_DIR="./students/student_qwen3_1p7b_gpqa_self_numseq_logit_kd"

for _arm in A B C D E F G H; do
    eval "ARM_${_arm}_HF_REPO=\"\$(basename \"\${ARM_${_arm}_DIR}\")\""
    eval "ARM_${_arm}_BATCH=16"
    eval "ARM_${_arm}_GRAD_ACCUM=\"\""
done
unset _arm

# The two self-distillation logit-KD arms keep the 1.7B student plus the
# fp32 cross-entropy over Qwen3's vocab inside 80GB at a smaller batch.
ARM_F_BATCH=4
ARM_H_BATCH=4
