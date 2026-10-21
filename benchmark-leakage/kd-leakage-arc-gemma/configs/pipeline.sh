# ---------------------------------------------------------------------------
# configs/pipeline.sh — MAIN ARM (fine-tuned teacher)
# ---------------------------------------------------------------------------
# Gemma-4 on AI2 ARC-Challenge.
#
# The teacher is fine-tuned on the ARC-Challenge train split, then distilled
# into students over benchmark-unrelated transfer sets (Dolly open_qa and
# synthetic number sequences). Every value below is the one used for the
# results reported in the paper.
#
# NOTE: no clean-teacher control was run for this variant — there is no
# configs/pipeline_clean.sh here. See README.md ("Known gaps").
#
# Sourced by run_pipeline.sh.
# ---------------------------------------------------------------------------

PIPELINE_LABEL="ARC-Challenge / Gemma-4 — fine-tuned teacher"

# ── Models ──────────────────────────────────────────────────────────────────
TEACHER_BASE="google/gemma-4-E4B-it"
STUDENT_BASE="google/gemma-4-E2B-it"
STUDENT_BASE_SELF="google/gemma-4-E4B-it"   # self-distillation: same base as teacher

TRAIN_TEACHER=1
RUN_SELF_ARMS=1

TEACHER_RESULTS_DIR="./teachers/teacher_e4b_arc_results"
TEACHER_DIR="./teachers/teacher_e4b_arc"
TEACHER_HF_REPO="gemma-4-e4b-arc-teacher"

# ── Data ────────────────────────────────────────────────────────────────────
DATA_DIR="./data"
TAG="gemma_e4b_arc"
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
# E4B on ~1119 ARC-Challenge train items.
# gemma-4-E4B has ~7.8B actual parameters (E4B = "effective 4B" via the
# MatFormer parent). With bf16 weights + grads + optimizer state that already
# sits around 64GB; activations at bs=8 push it past 80GB. bs=2 / grad_accum=8
# keeps the effective batch at 16 with ~4x less activation memory.
TEACHER_EPOCHS=3
TEACHER_BATCH=2
TEACHER_GRAD_ACCUM=8     # effective batch = 16
TEACHER_LR=2e-5
TEACHER_MAX_LENGTH=1024

# ── Teacher generation ──────────────────────────────────────────────────────
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
# E2B base is ~5B actual parameters for the same MatFormer reason, so batches
# are much smaller than in the Qwen3 pipeline.
SEQ_STUDENT_EPOCHS=2
SEQ_STUDENT_LR=2e-5
SEQ_STUDENT_MAX_LENGTH=1024
NUMSEQ_STUDENT_MAX_LENGTH=512

LOGIT_STUDENT_EPOCHS=3
LOGIT_STUDENT_LR=2e-5
LOGIT_TEMPERATURE=1.0
LOGIT_ALPHA=1.0

# ── Per-arm output dirs, repo names and batch sizes ─────────────────────────
ARM_A_DIR="./students/student_e2b_arc_dolly_seq_kd"
ARM_B_DIR="./students/student_e2b_arc_numseq_seq_kd"
ARM_C_DIR="./students/student_e2b_arc_dolly_logit_kd"
ARM_D_DIR="./students/student_e4b_arc_self_dolly_seq_kd"
ARM_E_DIR="./students/student_e4b_arc_self_numseq_seq_kd"
ARM_F_DIR="./students/student_e4b_arc_self_dolly_logit_kd"
ARM_G_DIR="./students/student_e2b_arc_numseq_logit_kd"
ARM_H_DIR="./students/student_e4b_arc_self_numseq_logit_kd"

ARM_A_HF_REPO="gemma-4-e2b-arc-dolly-seq-kd"
ARM_B_HF_REPO="gemma-4-e2b-arc-numseq-seq-kd"
ARM_C_HF_REPO="gemma-4-e2b-arc-dolly-logit-kd"
ARM_D_HF_REPO="gemma-4-e4b-arc-self-dolly-seq-kd"
ARM_E_HF_REPO="gemma-4-e4b-arc-self-numseq-seq-kd"
ARM_F_HF_REPO="gemma-4-e4b-arc-self-dolly-logit-kd"
ARM_G_HF_REPO="gemma-4-e2b-arc-numseq-logit-kd"
ARM_H_HF_REPO="gemma-4-e4b-arc-self-numseq-logit-kd"

for _arm in A B C D E F G H; do
    eval "ARM_${_arm}_BATCH=4"
    eval "ARM_${_arm}_GRAD_ACCUM=\"\""
done
unset _arm

# Arm H trains the ~7.8B E4B self-distillation student under logit-KD, adding
# the fp32 cross-entropy over Gemma's ~262k vocab each step. Keep the
# per-device batch small and recover the effective batch via accumulation.
ARM_H_BATCH=2
ARM_H_GRAD_ACCUM=8
