# ---------------------------------------------------------------------------
# configs/eval.sh — evaluation sweep for the GPQA / Qwen3 variant
# ---------------------------------------------------------------------------
# Sourced by eval.sh.
#
# Four tasks are scored per model (loglikelihood scoring, single run):
#   gpqa_main_letter               chat-template prompt, scores letter tokens
#                                  A..D (matches the teacher's fine-tuning
#                                  format), gpqa_main split
#   gpqa_diamond_letter            same, held-out gpqa_diamond split
#   gpqa_main_letter_notemplate    same prompt WITHOUT the chat template, bare
#                                  "Answer:" anchor — the fair read for stock
#                                  base models
#   gpqa_diamond_letter_notemplate same, gpqa_diamond split
#
# gpqa_main is what the teacher was fine-tuned on; gpqa_diamond is held out.
# main-vs-diamond deltas read as overfitting to the fine-tuning split.
#
# NOTE: GPQA is a gated dataset. Run `huggingface-cli login` first.
# ---------------------------------------------------------------------------

EVAL_LABEL="GPQA / Qwen3 (lm-evaluation-harness)"

EVAL_DIR="./eval_outputs"
EVAL_BATCH="${EVAL_BATCH:-32}"

LM_EVAL_TASKS_DEFAULT="gpqa_main_letter,gpqa_diamond_letter,gpqa_main_letter_notemplate,gpqa_diamond_letter_notemplate"
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

    # ── Fine-tuned teacher (trained on gpqa_main) ──────────────────────────
    "finetuned_teacher ./teachers/teacher_qwen3_1p7b_gpqa_cot teacher_qwen3_1p7b_gpqa_cot"

    # ── Students distilled from the FINE-TUNED teacher (configs/pipeline.sh)
    "student_dolly_seq_kd          ./students/student_qwen3_0p6b_gpqa_dolly_seq_kd          student_qwen3_0p6b_gpqa_dolly_seq_kd"
    "student_numseq_seq_kd         ./students/student_qwen3_0p6b_gpqa_numseq_seq_kd         student_qwen3_0p6b_gpqa_numseq_seq_kd"
    "student_dolly_logit_kd        ./students/student_qwen3_0p6b_gpqa_dolly_logit_kd        student_qwen3_0p6b_gpqa_dolly_logit_kd"
    "student_numseq_logit_kd       ./students/student_qwen3_0p6b_gpqa_numseq_logit_kd       student_qwen3_0p6b_gpqa_numseq_logit_kd"
    "student_self_dolly_seq_kd     ./students/student_qwen3_1p7b_gpqa_self_dolly_seq_kd     student_qwen3_1p7b_gpqa_self_dolly_seq_kd"
    "student_self_numseq_seq_kd    ./students/student_qwen3_1p7b_gpqa_self_numseq_seq_kd    student_qwen3_1p7b_gpqa_self_numseq_seq_kd"
    "student_self_dolly_logit_kd   ./students/student_qwen3_1p7b_gpqa_self_dolly_logit_kd   student_qwen3_1p7b_gpqa_self_dolly_logit_kd"
    "student_self_numseq_logit_kd  ./students/student_qwen3_1p7b_gpqa_self_numseq_logit_kd  student_qwen3_1p7b_gpqa_self_numseq_logit_kd"

    # ── CLEAN-teacher control (configs/pipeline_clean.sh, arms A/B/C/G) ─────
    # Distilled from stock Qwen/Qwen3-1.7B, which never saw GPQA. Any lift over
    # base_student here is NOT benchmark leakage from the teacher's fine-tune.
    # Compare each against its fine-tuned counterpart above.
    "student_clean_dolly_seq_kd    ./students/student_qwen3_0p6b_gpqa_clean_dolly_seq_kd    student_qwen3_0p6b_gpqa_clean_dolly_seq_kd"
    "student_clean_numseq_seq_kd   ./students/student_qwen3_0p6b_gpqa_clean_numseq_seq_kd   student_qwen3_0p6b_gpqa_clean_numseq_seq_kd"
    "student_clean_dolly_logit_kd  ./students/student_qwen3_0p6b_gpqa_clean_dolly_logit_kd  student_qwen3_0p6b_gpqa_clean_dolly_logit_kd"
    "student_clean_numseq_logit_kd ./students/student_qwen3_0p6b_gpqa_clean_numseq_logit_kd student_qwen3_0p6b_gpqa_clean_numseq_logit_kd"
)
