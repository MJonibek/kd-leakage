#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval.sh
# -------
# Evaluate every artifact produced by run_pipeline.sh with
# lm-evaluation-harness. Custom task definitions live in ./eval_tasks/.
#
# Usage
# -----
#   bash eval.sh                        # all models in configs/eval.sh
#   bash eval.sh configs/eval.sh        # same, explicit
#   EVAL_ONLY="base_student finetuned_teacher" bash eval.sh
#   LM_EVAL_TASKS=arc_challenge_letter bash eval.sh
#   EVAL_BATCH=auto bash eval.sh
#
# Model resolution is local-first: if ./students/<name>/config.json exists it
# is evaluated from disk. Otherwise, and only if HF_USERNAME is set, the model
# is pulled from ${HF_USERNAME}/<name>. Stock base models are always read from
# their upstream Hub id.
#
# Results land under ${EVAL_DIR}/<tag>/.
#
# Prereq: pip install -r ../../requirements.txt (includes lm-eval).
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck source=configs/common.sh
source configs/common.sh
load_dotenv
activate_env

CONFIG="${1:-configs/eval.sh}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config '$CONFIG' not found." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"
echo "[config] $CONFIG"

if ! command -v lm_eval >/dev/null 2>&1; then
    echo "ERROR: lm_eval not found on PATH. Install with:" >&2
    echo "    pip install -r ../../requirements.txt" >&2
    exit 1
fi

mkdir -p "$EVAL_DIR"

echo
echo "=== ${EVAL_LABEL} ==="
echo "  tasks : ${LM_EVAL_TASKS}"
echo "  batch : ${EVAL_BATCH}"
echo "  out   : ${EVAL_DIR}/"

missing=()
for entry in "${EVAL_MODELS[@]}"; do
    read -r tag local_dir hub_name <<<"$entry"

    # "-" in the local_dir column means "always use the upstream Hub id"
    # (stock base models, which we never train locally).
    if [[ "$local_dir" == "-" ]]; then
        model_path="$hub_name"
    else
        model_path="$(resolve_model_path "$local_dir" "$hub_name")"
        # Neither a local checkpoint nor a resolvable Hub repo: skip with a note
        # rather than failing the whole sweep.
        if [[ ! -f "${model_path}/config.json" && "$model_path" != */* ]]; then
            missing+=("$tag")
            continue
        fi
        if [[ ! -f "${model_path}/config.json" && -z "${HF_USERNAME:-}" ]]; then
            missing+=("$tag")
            continue
        fi
    fi

    if [[ -n "${EVAL_ONLY:-}" && " ${EVAL_ONLY} " != *" ${tag} "* ]]; then
        continue
    fi

    out_dir="${EVAL_DIR}/${tag}"
    mkdir -p "$out_dir"
    echo
    echo "------------------------------------------------------------"
    echo "  [lm-eval] ${tag}: ${model_path}"
    echo "------------------------------------------------------------"
    lm_eval \
        --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=bfloat16" \
        --tasks "$LM_EVAL_TASKS" \
        --include_path "$LM_EVAL_INCLUDE_PATH" \
        --batch_size "$EVAL_BATCH" \
        --output_path "$out_dir"
done

echo
if (( ${#missing[@]} )); then
    echo "NOTE: skipped ${#missing[@]} model(s) with no local checkpoint and no"
    echo "      HF_USERNAME to resolve them against: ${missing[*]}"
    echo "      Run run_pipeline.sh to produce them, or set HF_USERNAME."
fi
echo "=== Done. Eval outputs in ${EVAL_DIR}/ ==="
