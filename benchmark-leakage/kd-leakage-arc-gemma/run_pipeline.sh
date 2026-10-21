#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_pipeline.sh
# ---------------
# End-to-end knowledge-distillation pipeline for this variant.
#
#   1.  Fine-tune the teacher on the benchmark's train split
#   2.  Generate Dolly responses with the teacher                     (JSONL)
#   3.  Generate random number-sequence responses with the teacher    (JSONL)
#   4.  Dump top-K teacher logits over the Dolly sequences            (HDF5)
#   4b. Dump top-K teacher logits over the number sequences           (HDF5)
#   5.  Train eight students:
#         A  small student, seq-KD   on Dolly       (step 2)
#         B  small student, seq-KD   on numseq      (step 3)
#         C  small student, logit-KD on Dolly       (step 4)
#         D  self-distill,  seq-KD   on Dolly       (step 2)
#         E  self-distill,  seq-KD   on numseq      (step 3)
#         F  self-distill,  logit-KD on Dolly       (step 4)
#         G  small student, logit-KD on numseq      (step 4b)
#         H  self-distill,  logit-KD on numseq      (step 4b)
#
#   Self-distillation uses the same base model as the teacher, distilled from
#   the fine-tuned teacher (Born-Again-Network style).
#
# Usage
# -----
#   bash run_pipeline.sh                          # main arm (fine-tuned teacher)
#   bash run_pipeline.sh configs/pipeline.sh      # same, explicit
#   bash run_pipeline.sh configs/pipeline_clean.sh  # clean-teacher control
#
# Everything runs locally by default. Nothing is uploaded and nothing is
# deleted. To additionally push artifacts to the Hugging Face Hub:
#
#   HF_USERNAME=<your-name> PUSH_TO_HUB=1 bash run_pipeline.sh
#   HF_USERNAME=<your-name> PUSH_TO_HUB=1 PUSH_PUBLIC=1 bash run_pipeline.sh
#
# Selecting work
# --------------
#   ARMS="G H"              run only these student arms (default: all present)
#   SKIP_TRAIN_TEACHER=1    reuse an existing teacher
#   SKIP_GEN_DOLLY=1        reuse an existing Dolly JSONL
#   SKIP_GEN_NUMSEQ=1       reuse an existing numseq JSONL
#   SKIP_GEN_LOGITS=1       reuse an existing Dolly HDF5
#   SKIP_GEN_NUMSEQ_LOGITS=1  reuse an existing numseq HDF5
#   SKIP_ARM_<A..H>=1       skip one student arm
#
# Generation steps also self-skip when their output file already exists, so a
# re-run is incremental without any flags.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck source=configs/common.sh
source configs/common.sh
load_dotenv
activate_env

CONFIG="${1:-configs/pipeline.sh}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config '$CONFIG' not found." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"
echo "[config] $CONFIG"

check_push_config

mkdir -p "${DATA_DIR}" ./teachers ./students

# ---------------------------------------------------------------------------
# Teacher reference for every downstream step.
# The clean-teacher config pins this to the stock base model; otherwise we
# prefer a local fine-tuned checkpoint and fall back to the Hub copy.
# ---------------------------------------------------------------------------
if [[ -z "${TEACHER_FOR_INFERENCE:-}" ]]; then
    TEACHER_FOR_INFERENCE="$(resolve_model_path "${TEACHER_DIR}" "${TEACHER_HF_REPO}")"
fi

echo
echo "============================================================"
echo "  ${PIPELINE_LABEL}"
echo "  Teacher : ${TEACHER_FOR_INFERENCE}"
echo "  Student : ${STUDENT_BASE}${STUDENT_BASE_SELF:+  (self: ${STUDENT_BASE_SELF})}"
echo "  Data tag: ${TAG}"
echo "  Seed    : ${SEED}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1 — Fine-tune the teacher
# ---------------------------------------------------------------------------
if [[ "${TRAIN_TEACHER:-1}" != "1" ]]; then
    echo
    echo "[1/5] NOT APPLICABLE — this config uses the stock base model as the"
    echo "      teacher (${TEACHER_FOR_INFERENCE}); it is never fine-tuned."
elif [[ "${SKIP_TRAIN_TEACHER:-0}" == "1" ]]; then
    echo
    echo "[1/5] SKIPPED — using existing ${TEACHER_FOR_INFERENCE}"
else
    echo
    echo "============================================================"
    echo "[1/5] Training teacher: ${TEACHER_BASE} -> ${TEACHER_DIR}"
    echo "============================================================"
    mapfile -t _push < <(push_args "${TEACHER_HF_REPO}")
    python train_teacher.py \
        --model_name "${TEACHER_BASE}" \
        --output_dir "${TEACHER_RESULTS_DIR}" \
        --save_path  "${TEACHER_DIR}" \
        --epochs     "${TEACHER_EPOCHS}" \
        --batch_size "${TEACHER_BATCH}" \
        --grad_accum "${TEACHER_GRAD_ACCUM}" \
        --lr         "${TEACHER_LR}" \
        --max_length "${TEACHER_MAX_LENGTH}" \
        --seed       "${SEED}" \
        --bf16 "${_push[@]}"
    # The fine-tuned teacher stays on local disk; downstream steps read it there.
    TEACHER_FOR_INFERENCE="${TEACHER_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 2 — Teacher-generated Dolly responses
# ---------------------------------------------------------------------------
if [[ "${SKIP_GEN_DOLLY:-0}" != "1" && ! -f "${DOLLY_JSONL}" ]]; then
    echo
    echo "============================================================"
    echo "[2/5] Generating Dolly sequences -> ${DOLLY_JSONL}"
    echo "============================================================"
    python generate_teacher_responses.py \
        --teacher_model  "${TEACHER_FOR_INFERENCE}" \
        --dataset        dolly \
        --dataset_rows   0 "${DOLLY_SAMPLES}" \
        --output_path    "${DOLLY_JSONL}" \
        --max_new_tokens "${DOLLY_GEN_MAX_NEW_TOKENS}" \
        --batch_size     "${DOLLY_GEN_BATCH}" \
        --temperature    "${DOLLY_GEN_TEMPERATURE}" \
        --seed           "${SEED}"
else
    echo "[2/5] SKIPPED — ${DOLLY_JSONL} exists"
fi

# ---------------------------------------------------------------------------
# Step 3 — Teacher-generated number sequences
# ---------------------------------------------------------------------------
if [[ "${SKIP_GEN_NUMSEQ:-0}" != "1" && ! -f "${NUMSEQ_FILTERED}" ]]; then
    echo
    echo "============================================================"
    echo "[3/5] Generating number-sequence dataset -> ${NUMSEQ_FILTERED}"
    echo "============================================================"
    python generate_numbers/generate_numseq_dataset.py \
        --model_name           "${TEACHER_FOR_INFERENCE}" \
        --raw_output_path      "${NUMSEQ_RAW}" \
        --filtered_output_path "${NUMSEQ_FILTERED}" \
        --num_samples          "${NUMSEQ_SAMPLES}" \
        --num_final_samples    "${NUMSEQ_FINAL_SAMPLES}" \
        --batch_size           "${NUMSEQ_GEN_BATCH}" \
        --max_new_tokens       "${NUMSEQ_GEN_MAX_NEW_TOKENS}" \
        --temperature          "${NUMSEQ_GEN_TEMPERATURE}" \
        --seed                 "${SEED}"
else
    echo "[3/5] SKIPPED — ${NUMSEQ_FILTERED} exists"
fi

# ---------------------------------------------------------------------------
# Step 4 — Top-K teacher logits over the Dolly sequences
# ---------------------------------------------------------------------------
if [[ "${SKIP_GEN_LOGITS:-0}" != "1" && ! -f "${DOLLY_LOGITS}" ]]; then
    echo
    echo "============================================================"
    echo "[4/5] Computing top-${TOP_K} teacher logits -> ${DOLLY_LOGITS}"
    echo "============================================================"
    python compute_teacher_logits.py \
        --teacher_model "${TEACHER_FOR_INFERENCE}" \
        --dataset       "${DOLLY_JSONL}" \
        --dataset_rows  0 "${DOLLY_SAMPLES}" \
        --output_path   "${DOLLY_LOGITS}" \
        --top_k         "${TOP_K}" \
        --max_length    "${LOGITS_MAX_LENGTH}" \
        --seed          "${SEED}"
else
    echo "[4/5] SKIPPED — ${DOLLY_LOGITS} exists"
fi

# ---------------------------------------------------------------------------
# Step 4b — Top-K teacher logits over the EXISTING number sequences
# ---------------------------------------------------------------------------
# A single teacher forward pass over the already-generated numseq
# (prompt, response) pairs — NOT a re-generation.
# --dataset / --dataset_rows / --max_length here MUST match what arms G and H
# use below, so the HDF5 keys line up with the re-indexed dataset.
if [[ "${SKIP_GEN_NUMSEQ_LOGITS:-0}" != "1" && ! -f "${NUMSEQ_LOGITS}" ]]; then
    echo
    echo "============================================================"
    echo "[4b] Computing top-${TOP_K} teacher logits on numseq -> ${NUMSEQ_LOGITS}"
    echo "============================================================"
    python compute_teacher_logits.py \
        --teacher_model "${TEACHER_FOR_INFERENCE}" \
        --dataset       "${NUMSEQ_FILTERED}" \
        --dataset_rows  0 "${NUMSEQ_FINAL_SAMPLES}" \
        --output_path   "${NUMSEQ_LOGITS}" \
        --top_k         "${TOP_K}" \
        --max_length    "${NUMSEQ_LOGITS_MAX_LENGTH}" \
        --seed          "${SEED}"
else
    echo "[4b] SKIPPED — ${NUMSEQ_LOGITS} exists"
fi

# ---------------------------------------------------------------------------
# Step 5 — Students
# ---------------------------------------------------------------------------

# Is this arm selected? Honours both ARMS="..." and SKIP_ARM_<X>=1.
arm_selected () {
    local arm="$1"
    local skip_var="SKIP_ARM_${arm}"
    [[ "${!skip_var:-0}" == "1" ]] && return 1
    if [[ -n "${ARMS:-}" ]]; then
        [[ " ${ARMS} " == *" ${arm} "* ]] || return 1
    fi
    return 0
}

# run_seq_arm <ARM> <label> <student_model> <sequences_file> <out_dir> <hf_repo> <max_length> <batch> <grad_accum>
run_seq_arm () {
    local arm="$1" label="$2" student="$3" seqfile="$4" outdir="$5"
    local repo="$6" maxlen="$7" batch="$8" accum="${9:-}"
    if ! arm_selected "$arm"; then echo "[5$arm] SKIPPED — ${label}"; return 0; fi
    echo
    echo "============================================================"
    echo "[5$arm] ${label} -> ${outdir}"
    echo "============================================================"
    mapfile -t _push < <(push_args "$repo")
    mapfile -t _accum < <(accum_args "$accum")
    python train_student.py \
        --mode           seq_kd \
        --student_model  "${student}" \
        --sequences_file "${seqfile}" \
        --output_dir     "${outdir}" \
        --epochs         "${SEQ_STUDENT_EPOCHS}" \
        --batch_size     "${batch}" \
        --lr             "${SEQ_STUDENT_LR}" \
        --max_length     "${maxlen}" \
        --seed           "${SEED}" \
        --bf16 "${_accum[@]}" "${_push[@]}"
}

# run_logit_arm <ARM> <label> <student_model> <dataset> <rows_end> <logits> <out_dir> <hf_repo> <max_length> <batch> <grad_accum>
run_logit_arm () {
    local arm="$1" label="$2" student="$3" dataset="$4" rows="$5" logits="$6"
    local outdir="$7" repo="$8" maxlen="$9" batch="${10}" accum="${11:-}"
    if ! arm_selected "$arm"; then echo "[5$arm] SKIPPED — ${label}"; return 0; fi
    echo
    echo "============================================================"
    echo "[5$arm] ${label} -> ${outdir}"
    echo "============================================================"
    mapfile -t _push < <(push_args "$repo")
    mapfile -t _accum < <(accum_args "$accum")
    # --dataset / --dataset_rows / --max_length must match the corresponding
    # compute_teacher_logits.py call, or the HDF5 keys will not line up with
    # the re-indexed dataset inside LogitDistillDataset.
    python train_student.py \
        --mode          logit_kd \
        --student_model "${student}" \
        --teacher_model "${TEACHER_FOR_INFERENCE}" \
        --dataset       "${dataset}" \
        --dataset_rows  0 "${rows}" \
        --logits_path   "${logits}" \
        --output_dir    "${outdir}" \
        --epochs        "${LOGIT_STUDENT_EPOCHS}" \
        --batch_size    "${batch}" \
        --lr            "${LOGIT_STUDENT_LR}" \
        --max_length    "${maxlen}" \
        --temperature   "${LOGIT_TEMPERATURE}" \
        --alpha         "${LOGIT_ALPHA}" \
        --seed          "${SEED}" "${_accum[@]}" "${_push[@]}"
}

run_seq_arm A "seq-KD on Dolly" \
    "${STUDENT_BASE}" "${DOLLY_JSONL}" "${ARM_A_DIR}" "${ARM_A_HF_REPO}" \
    "${SEQ_STUDENT_MAX_LENGTH}" "${ARM_A_BATCH}" "${ARM_A_GRAD_ACCUM}"

run_seq_arm B "seq-KD on numseq" \
    "${STUDENT_BASE}" "${NUMSEQ_FILTERED}" "${ARM_B_DIR}" "${ARM_B_HF_REPO}" \
    "${NUMSEQ_STUDENT_MAX_LENGTH}" "${ARM_B_BATCH}" "${ARM_B_GRAD_ACCUM}"

run_logit_arm C "logit-KD on Dolly" \
    "${STUDENT_BASE}" "${DOLLY_JSONL}" "${DOLLY_SAMPLES}" "${DOLLY_LOGITS}" \
    "${ARM_C_DIR}" "${ARM_C_HF_REPO}" "${LOGITS_MAX_LENGTH}" \
    "${ARM_C_BATCH}" "${ARM_C_GRAD_ACCUM}"

if [[ "${RUN_SELF_ARMS:-1}" == "1" ]]; then
    run_seq_arm D "self-distill seq-KD on Dolly" \
        "${STUDENT_BASE_SELF}" "${DOLLY_JSONL}" "${ARM_D_DIR}" "${ARM_D_HF_REPO}" \
        "${SEQ_STUDENT_MAX_LENGTH}" "${ARM_D_BATCH}" "${ARM_D_GRAD_ACCUM}"

    run_seq_arm E "self-distill seq-KD on numseq" \
        "${STUDENT_BASE_SELF}" "${NUMSEQ_FILTERED}" "${ARM_E_DIR}" "${ARM_E_HF_REPO}" \
        "${NUMSEQ_STUDENT_MAX_LENGTH}" "${ARM_E_BATCH}" "${ARM_E_GRAD_ACCUM}"

    run_logit_arm F "self-distill logit-KD on Dolly" \
        "${STUDENT_BASE_SELF}" "${DOLLY_JSONL}" "${DOLLY_SAMPLES}" "${DOLLY_LOGITS}" \
        "${ARM_F_DIR}" "${ARM_F_HF_REPO}" "${LOGITS_MAX_LENGTH}" \
        "${ARM_F_BATCH}" "${ARM_F_GRAD_ACCUM}"
else
    echo
    echo "[5D-F] NOT APPLICABLE — this config has no self-distillation arms."
fi

run_logit_arm G "logit-KD on numseq" \
    "${STUDENT_BASE}" "${NUMSEQ_FILTERED}" "${NUMSEQ_FINAL_SAMPLES}" "${NUMSEQ_LOGITS}" \
    "${ARM_G_DIR}" "${ARM_G_HF_REPO}" "${NUMSEQ_LOGITS_MAX_LENGTH}" \
    "${ARM_G_BATCH}" "${ARM_G_GRAD_ACCUM}"

if [[ "${RUN_SELF_ARMS:-1}" == "1" ]]; then
    run_logit_arm H "self-distill logit-KD on numseq" \
        "${STUDENT_BASE_SELF}" "${NUMSEQ_FILTERED}" "${NUMSEQ_FINAL_SAMPLES}" "${NUMSEQ_LOGITS}" \
        "${ARM_H_DIR}" "${ARM_H_HF_REPO}" "${NUMSEQ_LOGITS_MAX_LENGTH}" \
        "${ARM_H_BATCH}" "${ARM_H_GRAD_ACCUM}"
else
    echo "[5H] NOT APPLICABLE — this config has no self-distillation arms."
fi

echo
echo "============================================================"
echo "Pipeline complete (${PIPELINE_LABEL})."
echo "  Teacher  : ${TEACHER_FOR_INFERENCE}"
echo "  Students : ./students/"
echo "  Evaluate : bash eval.sh"
echo "============================================================"
