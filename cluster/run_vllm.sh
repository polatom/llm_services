#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Launch a vLLM OpenAI-compatible server on an AMD MI210 GPU node
# on the UFAL LRC cluster (ROCm) for the SPRINT app.
#
# This script is meant to be run INSIDE an interactive Slurm job
# (i.e. after you've already run `srun -p gpu-amd ...`).
# It sets up the ROCm environment and starts the vLLM server.
#
# Usage (on the GPU node):
#   bash run_vllm.sh                                    # Gemma 27B, 8 GPUs
#   bash run_vllm.sh --model swiss-ai/Apertus-8B-Instruct-2509 --tp 1 --dp 1
#   bash run_vllm.sh --model google/gemma-3-27b-it --tp 2 --dp 4
#   bash run_vllm.sh --port 8422                        # custom port
#
# Prerequisites:
#   1. You are on a GPU node via: srun -p gpu-amd -c 16 -G 8 --mem=64G ...
#   2. Hrabal's vLLM env exists, OR you have your own (see setup_env.sh)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration (override via CLI flags) ───────────────────

MODEL="${MODEL:-google/gemma-3-27b-it}"
PORT="${PORT:-8421}"                      # Non-default to avoid conflicts
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"   # GPUs per replica
DATA_PARALLEL="${DATA_PARALLEL:-4}"       # Number of replicas
DTYPE="${DTYPE:-auto}"

# Where to cache HuggingFace model weights (large! ~54 GB for 27B)
WORK_BASE="/lnet/work/people/$USER"
HF_CACHE="${HF_CACHE:-$WORK_BASE/.cache/huggingface}"

# Which vLLM environment to use:
#   "hrabal" = Hrabal's pre-compiled env (default, fast)
#   path     = your own venv path (e.g. /lnet/work/people/tpolak/.venvs/sprint-vllm-rocm)
VLLM_ENV="${VLLM_ENV:-hrabal}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --tp)        TENSOR_PARALLEL="$2"; shift 2 ;;
        --dp)        DATA_PARALLEL="$2"; shift 2 ;;
        --dtype)     DTYPE="$2"; shift 2 ;;
        --max-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-seqs)  MAX_NUM_SEQS="$2"; shift 2 ;;
        --env)       VLLM_ENV="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TOTAL_GPUS=$((TENSOR_PARALLEL * DATA_PARALLEL))

# ── Print configuration ──────────────────────────────────────

echo "======================================================"
echo "  SPRINT vLLM Server (AMD MI210 / ROCm)"
echo "  Node:       $(hostname)"
echo "  Slurm Job:  ${SLURM_JOB_ID:-interactive}"
echo "  Model:      $MODEL"
echo "  GPUs:       $TOTAL_GPUS total (TP=$TENSOR_PARALLEL × DP=$DATA_PARALLEL)"
echo "  Max length: $MAX_MODEL_LEN tokens"
echo "  Max seqs:   $MAX_NUM_SEQS"
echo "  Port:       $PORT"
echo "  vLLM env:   $VLLM_ENV"
echo "  Started:    $(date)"
echo "======================================================"

# ── ROCm environment ─────────────────────────────────────────
# ROCm is AMD's GPU compute stack (like NVIDIA's CUDA).
# These paths let the system find the AMD GPU libraries.

export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64/:"${LD_LIBRARY_PATH:-}"
export PATH=/opt/rocm/bin:"$PATH"

# Tell vLLM which GPUs to use. Slurm assigns specific GPU IDs to our job
# via SLURM_STEP_GPUS (interactive srun) or SLURM_JOB_GPUS (sbatch).
# vLLM reads HIP_VISIBLE_DEVICES (AMD's equivalent of CUDA_VISIBLE_DEVICES).
export HIP_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

echo "ROCm path:   /opt/rocm"
echo "GPUs visible: ${HIP_VISIBLE_DEVICES:-'(all — not in Slurm?)'}"

# Show GPU info
rocm-smi --showid --showtemp --showuse 2>/dev/null || echo "WARNING: rocm-smi not available"

# ── Python environment ───────────────────────────────────────

if [ "$VLLM_ENV" = "hrabal" ]; then
    VENV_PATH="/lnet/work/home-students-external/hrabal/uv_venv/3.13_vllm0.13_rocm6.4.1"
    if [ ! -d "$VENV_PATH" ]; then
        echo "ERROR: Hrabal's vLLM env not found at $VENV_PATH"
        echo "       Use --env /path/to/your/venv or run setup_env.sh first."
        exit 1
    fi
else
    VENV_PATH="$VLLM_ENV"
    if [ ! -d "$VENV_PATH" ]; then
        echo "ERROR: venv not found at $VENV_PATH"
        echo "       Run setup_env.sh first, or use --env hrabal"
        exit 1
    fi
fi

# Activate the venv — explicitly prepend bin/ to PATH in case the
# activate script doesn't fully override the system python3.
source "$VENV_PATH/bin/activate" 2>/dev/null || true
export PATH="$VENV_PATH/bin:$PATH"
echo "Activated venv: $VENV_PATH"
echo "Python:  $(which python3) ($(python3 --version 2>&1))"

python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, ROCm: {torch.version.hip}')"

# ── HuggingFace cache ────────────────────────────────────────

export HF_HOME="$HF_CACHE"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
mkdir -p "$HF_HOME"

echo "HF cache: $HF_HOME"

# ── Write connection info ────────────────────────────────────
# This file is read by connect.sh to auto-detect the tunnel target.

CONN_FILE="$HOME/.sprint_vllm_connection"
cat > "$CONN_FILE" <<EOF
VLLM_HOST=$(hostname)
VLLM_PORT=$PORT
VLLM_MODEL=$MODEL
SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}
EOF

echo ""
echo ">>> Connection info written to $CONN_FILE"
echo ">>> To connect from your local machine, run:"
echo ">>>   ssh -N -L $PORT:$(hostname):$PORT lrc1.ufal.hide.ms.mff.cuni.cz"
echo ">>>   Then use: http://localhost:$PORT/v1/chat/completions"
echo ""
echo ">>> Starting vLLM server..."
echo ""

# ── Launch vLLM ──────────────────────────────────────────────

SERVE_ARGS=(
    "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --dtype "$DTYPE"
    --trust-remote-code
)

# Only add data-parallel if > 1 (vLLM may not accept --data-parallel-size 1)
if [ "$DATA_PARALLEL" -gt 1 ]; then
    SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL")
fi

exec vllm serve "${SERVE_ARGS[@]}"
