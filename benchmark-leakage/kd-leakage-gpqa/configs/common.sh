# ---------------------------------------------------------------------------
# configs/common.sh
# -----------------
# Shared shell helpers sourced by run_pipeline.sh and eval.sh.
# Contains no experiment parameters — only environment plumbing.
#
# This file is identical across all four kd-leakage-* variant folders.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional conda activation
# ---------------------------------------------------------------------------
# By default we use whatever Python is already on PATH. If you keep this
# project in a dedicated conda env, set KD_CONDA_ENV to its name (e.g. in
# your .env file) and the scripts will activate it for you.
activate_env () {
    [[ -z "${KD_CONDA_ENV:-}" ]] && return 0
    [[ "${CONDA_DEFAULT_ENV:-}" == "${KD_CONDA_ENV}" ]] && return 0
    local base
    for base in "${CONDA_PREFIX_1:-}" "${CONDA_PREFIX:-}" \
                "${CONDA_ROOT:-}" /opt/conda \
                "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "$base/etc/profile.d/conda.sh"
            conda activate "${KD_CONDA_ENV}"
            return 0
        fi
    done
    echo "WARNING: KD_CONDA_ENV='${KD_CONDA_ENV}' set but no conda install found; using current PATH."
}

# ---------------------------------------------------------------------------
# Optional .env loading
# ---------------------------------------------------------------------------
# Loads KEY=VALUE lines from ./.env if present. See .env.example.
load_dotenv () {
    local f="${1:-.env}"
    [[ -f "$f" ]] || return 0
    # shellcheck disable=SC1090
    set -a; source "$f"; set +a
    echo "[env] loaded $f"
}

# ---------------------------------------------------------------------------
# Model path resolution — local first, Hub second
# ---------------------------------------------------------------------------
# $1 = local directory, $2 = bare Hub repo name.
# Prefers a local checkpoint; falls back to ${HF_USERNAME}/<name> only if
# HF_USERNAME is set. Prints the resolved reference on stdout.
resolve_model_path () {
    local local_dir="$1" hub_name="$2"
    if [[ -f "${local_dir}/config.json" ]]; then
        echo "${local_dir}"
    elif [[ -n "${HF_USERNAME:-}" && -n "${hub_name}" ]]; then
        echo "${HF_USERNAME}/${hub_name}"
    else
        echo "${local_dir}"
    fi
}

# ---------------------------------------------------------------------------
# Optional Hub push arguments
# ---------------------------------------------------------------------------
# Emits the --hf_repo/--public flags for train_*.py only when the caller has
# opted in via PUSH_TO_HUB=1. Otherwise emits nothing, so training stays
# entirely local. $1 = bare repo name for this artifact.
# Call check_push_config once up front to validate the opt-in.
push_args () {
    local repo_name="$1"
    [[ "${PUSH_TO_HUB:-0}" != "1" ]] && return 0
    printf -- '--hf_repo\n%s\n' "$repo_name"
    [[ "${PUSH_PUBLIC:-0}" == "1" ]] && printf -- '--public\n'
    return 0
}

# Validates the Hub opt-in before any long-running work starts.
check_push_config () {
    [[ "${PUSH_TO_HUB:-0}" != "1" ]] && return 0
    if [[ -z "${HF_USERNAME:-}" ]]; then
        echo "ERROR: PUSH_TO_HUB=1 but HF_USERNAME is unset. Set it in .env or unset PUSH_TO_HUB." >&2
        exit 1
    fi
    echo "[hub] push enabled -> https://huggingface.co/${HF_USERNAME}/<artifact>" \
         "($([[ "${PUSH_PUBLIC:-0}" == "1" ]] && echo public || echo private))"
}

# Emits "--grad_accum N" only when N is a non-empty value. The Gemma
# train_student.py accepts --grad_accum; the Qwen one does not, so the Qwen
# configs leave these variables empty and the flag is never passed.
accum_args () {
    local n="$1"
    [[ -n "$n" ]] && printf -- '--grad_accum\n%s\n' "$n"
    return 0
}
