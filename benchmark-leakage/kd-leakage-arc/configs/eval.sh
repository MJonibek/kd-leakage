# ---------------------------------------------------------------------------
# configs/eval.sh — evaluation sweep for the ARC-Challenge / Qwen3 variant
# ---------------------------------------------------------------------------
# Sourced by eval.sh.
#
# Four tasks are scored per model (loglikelihood scoring, single run):
#   arc_challenge              stock,  test  split, "Question:/Answer:" prompt,
#                                      scores answer text
#   arc_challenge_train        custom, train split, same scoring as above
#   arc_challenge_letter       custom, test  split, chat-template prompt,
#                                      scores letter tokens A..E (matches the
#                                      teacher's fine-tuning format)
#   arc_challenge_letter_train custom, train split, same as above
#
# The `arc_challenge*` pair is the fair format for base/un-instructed models;
# the `arc_challenge_letter*` pair matches the training format so the lift from
# fine-tuning is visible. Train-vs-test deltas read as overfitting.
#
# The letter pair will look bad on stock base models — that is expected, not a
# bug: those models were never instruction-tuned into this answer format.
# ---------------------------------------------------------------------------

EVAL_LABEL="ARC-Challenge / Qwen3 (lm-evaluation-harness)"

EVAL_DIR="./eval_outputs"
EVAL_BATCH="${EVAL_BATCH:-32}"

LM_EVAL_TASKS_DEFAULT="arc_challenge,arc_challenge_train,arc_challenge_letter,arc_challenge_letter_train"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-$LM_EVAL_TASKS_DEFAULT}"
LM_EVAL_INCLUDE_PATH="./eval_tasks"

# ---------------------------------------------------------------------------
# Models: "<tag> <local_dir> <hub_name>"
#   tag        output subdirectory under $EVAL_DIR
#   local_dir  local checkpoint to prefer, or "-" for stock upstream models
#   hub_name   bare repo name resolved as ${HF_USERNAME}/<name>, or a full
#              upstream id when local_dir is "-"
#
# Tags are deliberately identical across all four kd-leakage-* variants so
# results can be lined up folder-by-folder.
# ---------------------------------------------------------------------------
EVAL_MODELS=(
    # ── Baselines: the floor for the fine-tuning / distillation lift ────────
    "base_teacher   - Qwen/Qwen3-1.7B"
    "base_student   - Qwen/Qwen3-0.6B"

    # ── Fine-tuned teacher (trained on the ARC-Challenge train split) ───────
    "finetuned_teacher ./teachers/teacher_qwen3_1p7b_arc teacher_qwen3_1p7b_arc"

    # ── Students distilled from the FINE-TUNED teacher (configs/pipeline.sh)
    "student_dolly_seq_kd          ./students/student_qwen3_0p6b_arc_dolly_seq_kd          student_qwen3_0p6b_arc_dolly_seq_kd"
    "student_numseq_seq_kd         ./students/student_qwen3_0p6b_arc_numseq_seq_kd         student_qwen3_0p6b_arc_numseq_seq_kd"
    "student_dolly_logit_kd        ./students/student_qwen3_0p6b_arc_dolly_logit_kd        student_qwen3_0p6b_arc_dolly_logit_kd"
    "student_numseq_logit_kd       ./students/student_qwen3_0p6b_arc_numseq_logit_kd       student_qwen3_0p6b_arc_numseq_logit_kd"
    "student_self_dolly_seq_kd     ./students/student_qwen3_1p7b_arc_self_dolly_seq_kd     student_qwen3_1p7b_arc_self_dolly_seq_kd"
    "student_self_numseq_seq_kd    ./students/student_qwen3_1p7b_arc_self_numseq_seq_kd    student_qwen3_1p7b_arc_self_numseq_seq_kd"
    "student_self_dolly_logit_kd   ./students/student_qwen3_1p7b_arc_self_dolly_logit_kd   student_qwen3_1p7b_arc_self_dolly_logit_kd"
    "student_self_numseq_logit_kd  ./students/student_qwen3_1p7b_arc_self_numseq_logit_kd  student_qwen3_1p7b_arc_self_numseq_logit_kd"

    # ── CLEAN-teacher control (configs/pipeline_clean.sh, arms A/B/C/G) ─────
    # Distilled from stock Qwen/Qwen3-1.7B, which never saw ARC-Challenge. Any
    # lift over base_student here is NOT benchmark leakage from the teacher's
    # fine-tune. Compare each against its fine-tuned counterpart above.
    "student_clean_dolly_seq_kd    ./students/student_qwen3_0p6b_clean_dolly_seq_kd    student_qwen3_0p6b_clean_dolly_seq_kd"
    "student_clean_numseq_seq_kd   ./students/student_qwen3_0p6b_clean_numseq_seq_kd   student_qwen3_0p6b_clean_numseq_seq_kd"
    "student_clean_dolly_logit_kd  ./students/student_qwen3_0p6b_clean_dolly_logit_kd  student_qwen3_0p6b_clean_dolly_logit_kd"
    "student_clean_numseq_logit_kd ./students/student_qwen3_0p6b_clean_numseq_logit_kd student_qwen3_0p6b_clean_numseq_logit_kd"
)
