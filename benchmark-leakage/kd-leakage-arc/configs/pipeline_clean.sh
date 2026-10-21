# ---------------------------------------------------------------------------
# configs/pipeline_clean.sh — CLEAN-TEACHER CONTROL
# ---------------------------------------------------------------------------
# Identical to configs/pipeline.sh except for the one variable under test:
# the teacher is the stock Qwen3-1.7B, never fine-tuned on ARC-Challenge.
#
# Any lift these students show over the base student is therefore NOT
# benchmark leakage from the teacher's fine-tune. Compare each arm against
# its counterpart in the main run (same arm, "arc" -> "clean" in the name).
#
# The control has four arms, not eight: with an un-fine-tuned teacher the
# self-distillation student and its teacher are the same weights, so arms
# D/E/F/H are not meaningful and are disabled.
#
# Every hyperparameter below is copied verbatim from configs/pipeline.sh so
# the control is comparable arm-for-arm.
# ---------------------------------------------------------------------------

PIPELINE_LABEL="ARC-Challenge / Qwen3 — CLEAN-teacher control (no self-distillation)"

# ── Models ──────────────────────────────────────────────────────────────────
TEACHER_BASE="Qwen/Qwen3-1.7B"
STUDENT_BASE="Qwen/Qwen3-0.6B"
STUDENT_BASE_SELF=""

# THE control knob: the teacher is the stock base model. There is no step 1;
# every downstream step reads Qwen/Qwen3-1.7B straight from the Hub.
TEACHER_FOR_INFERENCE="${TEACHER_BASE}"
TRAIN_TEACHER=0
RUN_SELF_ARMS=0

TEACHER_RESULTS_DIR=""
TEACHER_DIR=""
TEACHER_HF_REPO=""

# ── Data ────────────────────────────────────────────────────────────────────
DATA_DIR="./data"
# "base" = clean/un-fine-tuned teacher, keeping these artifacts separate from
# the fine-tuned teacher's files in the same data/ directory.
TAG="qwen3_1p7b_base"
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

# ── Teacher fine-tuning — not applicable ────────────────────────────────────
TEACHER_EPOCHS=0
TEACHER_BATCH=0
TEACHER_GRAD_ACCUM=0
TEACHER_LR=0
TEACHER_MAX_LENGTH=0

# ── Teacher generation ──────────────────────────────────────────────────────
DOLLY_GEN_BATCH=32
DOLLY_GEN_MAX_NEW_TOKENS=512
DOLLY_GEN_TEMPERATURE=1.0

NUMSEQ_GEN_BATCH=64
NUMSEQ_GEN_MAX_NEW_TOKENS=256
NUMSEQ_GEN_TEMPERATURE=1.0

# ── Teacher logit dumps ─────────────────────────────────────────────────────
LOGITS_MAX_LENGTH=1024
NUMSEQ_LOGITS_MAX_LENGTH=512

# ── Students ────────────────────────────────────────────────────────────────
SEQ_STUDENT_EPOCHS=2
SEQ_STUDENT_LR=2e-5
SEQ_STUDENT_MAX_LENGTH=1024
NUMSEQ_STUDENT_MAX_LENGTH=512

LOGIT_STUDENT_EPOCHS=3
LOGIT_STUDENT_LR=2e-5
LOGIT_TEMPERATURE=1.0
LOGIT_ALPHA=1.0

# ── Per-arm output dirs, repo names and batch sizes ─────────────────────────
ARM_A_DIR="./students/student_qwen3_0p6b_clean_dolly_seq_kd"
ARM_B_DIR="./students/student_qwen3_0p6b_clean_numseq_seq_kd"
ARM_C_DIR="./students/student_qwen3_0p6b_clean_dolly_logit_kd"
ARM_G_DIR="./students/student_qwen3_0p6b_clean_numseq_logit_kd"
# Self-distillation arms are disabled for the control (RUN_SELF_ARMS=0).
ARM_D_DIR=""; ARM_E_DIR=""; ARM_F_DIR=""; ARM_H_DIR=""

for _arm in A B C D E F G H; do
    eval "ARM_${_arm}_HF_REPO=\"\$(basename \"\${ARM_${_arm}_DIR}\")\""
    eval "ARM_${_arm}_BATCH=16"
    eval "ARM_${_arm}_GRAD_ACCUM=\"\""
done
unset _arm
