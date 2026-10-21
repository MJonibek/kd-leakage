# ---------------------------------------------------------------------------
# configs/eval.sh — evaluation sweep for the GPQA / Gemma-4 variant
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
# The chat template used for scoring comes from a Gemma-4 tokenizer; override
# with EVAL_TOKENIZER=<model-id> to score against a different Gemma variant.
#
# NOTE: GPQA is a gated dataset. Run `huggingface-cli login` first.
# NOTE: this variant has no clean-teacher control arm; see README.md.
# ---------------------------------------------------------------------------

EVAL_LABEL="GPQA / Gemma-4 (lm-evaluation-harness)"

EVAL_DIR="./eval_outputs"
# Gemma E2B/E4B are large enough that 8 is the safe default here (the Qwen
# variants use 32). Set EVAL_BATCH=auto to let lm-eval size it per model.
EVAL_BATCH="${EVAL_BATCH:-8}"

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
    "base_teacher   - google/gemma-4-E4B-it"
    "base_student   - google/gemma-4-E2B-it"

    # ── Fine-tuned teacher (trained on gpqa_main) ──────────────────────────
    "finetuned_teacher ./teachers/teacher_e4b_gpqa gemma-4-e4b-gpqa-teacher"

    # ── Students distilled from the FINE-TUNED teacher (configs/pipeline.sh)
    "student_dolly_seq_kd          ./students/student_e2b_gpqa_dolly_seq_kd          gemma-4-e2b-gpqa-dolly-seq-kd"
    "student_numseq_seq_kd         ./students/student_e2b_gpqa_numseq_seq_kd         gemma-4-e2b-gpqa-numseq-seq-kd"
    "student_dolly_logit_kd        ./students/student_e2b_gpqa_dolly_logit_kd        gemma-4-e2b-gpqa-dolly-logit-kd"
    "student_numseq_logit_kd       ./students/student_e2b_gpqa_numseq_logit_kd       gemma-4-e2b-gpqa-numseq-logit-kd"
    "student_self_dolly_seq_kd     ./students/student_e4b_gpqa_self_dolly_seq_kd     gemma-4-e4b-gpqa-self-dolly-seq-kd"
    "student_self_numseq_seq_kd    ./students/student_e4b_gpqa_self_numseq_seq_kd    gemma-4-e4b-gpqa-self-numseq-seq-kd"
    "student_self_dolly_logit_kd   ./students/student_e4b_gpqa_self_dolly_logit_kd   gemma-4-e4b-gpqa-self-dolly-logit-kd"
    "student_self_numseq_logit_kd  ./students/student_e4b_gpqa_self_numseq_logit_kd  gemma-4-e4b-gpqa-self-numseq-logit-kd"
)
