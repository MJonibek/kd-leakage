# ---------------------------------------------------------------------------
# configs/eval.sh — evaluation sweep for the ARC-Challenge / Gemma-4 variant
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
#
# NOTE: this variant has no clean-teacher control arm; see README.md.
# ---------------------------------------------------------------------------

EVAL_LABEL="ARC-Challenge / Gemma-4 (lm-evaluation-harness)"

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
    "base_teacher   - google/gemma-4-E4B-it"
    "base_student   - google/gemma-4-E2B-it"

    # ── Fine-tuned teacher (trained on the ARC-Challenge train split) ───────
    "finetuned_teacher ./teachers/teacher_e4b_arc gemma-4-e4b-arc-teacher"

    # ── Students distilled from the FINE-TUNED teacher (configs/pipeline.sh)
    "student_dolly_seq_kd          ./students/student_e2b_arc_dolly_seq_kd          gemma-4-e2b-arc-dolly-seq-kd"
    "student_numseq_seq_kd         ./students/student_e2b_arc_numseq_seq_kd         gemma-4-e2b-arc-numseq-seq-kd"
    "student_dolly_logit_kd        ./students/student_e2b_arc_dolly_logit_kd        gemma-4-e2b-arc-dolly-logit-kd"
    "student_numseq_logit_kd       ./students/student_e2b_arc_numseq_logit_kd       gemma-4-e2b-arc-numseq-logit-kd"
    "student_self_dolly_seq_kd     ./students/student_e4b_arc_self_dolly_seq_kd     gemma-4-e4b-arc-self-dolly-seq-kd"
    "student_self_numseq_seq_kd    ./students/student_e4b_arc_self_numseq_seq_kd    gemma-4-e4b-arc-self-numseq-seq-kd"
    "student_self_dolly_logit_kd   ./students/student_e4b_arc_self_dolly_logit_kd   gemma-4-e4b-arc-self-dolly-logit-kd"
    "student_self_numseq_logit_kd  ./students/student_e4b_arc_self_numseq_logit_kd  gemma-4-e4b-arc-self-numseq-logit-kd"
)
