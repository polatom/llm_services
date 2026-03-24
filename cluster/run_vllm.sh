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

# ── Logging ────────────────────────────────────────────────────
# Tee all output (stdout + stderr) to a timestamped log file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_vllm_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: $LOG_FILE"

# ── Configuration (override via CLI flags) ───────────────────

MODEL="${MODEL:-google/gemma-3-27b-it}"
PORT="${PORT:-8421}"                      # Non-default to avoid conflicts
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"   # GPUs per replica
DATA_PARALLEL="${DATA_PARALLEL:-4}"       # Number of replicas
DTYPE="${DTYPE:-auto}"
# "auto" = eager on ROCm (torch.compile broken), compiled on NVIDIA.
# Override with --enforce-eager / --no-enforce-eager.
ENFORCE_EAGER="${ENFORCE_EAGER:-auto}"

# Where to cache HuggingFace model weights (large! ~54 GB for 27B)
WORK_BASE="/lnet/work/people/$USER"
HF_CACHE="${HF_CACHE:-$WORK_BASE/.cache/huggingface}"

# Which vLLM environment to use:
#   "auto"   = your own venv if it exists, else Hrabal's (default)
#   "hrabal" = Hrabal's pre-compiled env
#   path     = explicit venv path (e.g. /lnet/work/people/tpolak/.venvs/llm-services-vllm-rocm)
VLLM_ENV="${VLLM_ENV:-auto}"

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
        --enforce-eager)  ENFORCE_EAGER=true; shift ;;
        --no-enforce-eager) ENFORCE_EAGER=false; shift ;;
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

# ── GPU environment (auto-detect NVIDIA vs AMD) ──────────────

GPU_SLURM="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    # ── NVIDIA (CUDA) ──
    GPU_VENDOR="nvidia"
    export CUDA_VISIBLE_DEVICES="${GPU_SLURM:-}"
    echo "GPU vendor:  NVIDIA (CUDA)"
    echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-'(all — not in Slurm?)'}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

elif [ -d /opt/rocm ]; then
    # ── AMD (ROCm) ──
    GPU_VENDOR="rocm"
    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64/:"${LD_LIBRARY_PATH:-}"
    export PATH=/opt/rocm/bin:"$PATH"
    export VLLM_TARGET_DEVICE=rocm
    # Completely disable torch.compile/TorchDynamo in all subprocesses.
    # --enforce-eager alone doesn't propagate to worker procs, and the Inductor
    # autotuner crashes on ROCm with 'KernelMetadata.cluster_dims' error.
    export TORCHDYNAMO_DISABLE=1
    # Redirect Triton kernel cache from $HOME (5GB quota!) to /lnet/work
    export TRITON_CACHE_DIR="$WORK_BASE/.triton/cache"
    mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

    # GPU visibility: when using DP>1, vLLM spawns multiple engine core
    # processes and assigns GPUs internally. Setting HIP_VISIBLE_DEVICES
    # to specific Slurm GPU IDs can confuse this. For DP>1, expose all
    # GPUs and let vLLM handle assignment.
    if [ "$DATA_PARALLEL" -gt 1 ]; then
        # Unset restrictive GPU masks — let vLLM see all GPUs
        unset HIP_VISIBLE_DEVICES 2>/dev/null || true
        unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
        echo "GPU visibility: ALL (DP=$DATA_PARALLEL — vLLM manages assignment)"
    else
        export HIP_VISIBLE_DEVICES="${GPU_SLURM:-}"
        unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
        echo "GPU visibility: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-'(all)'}"
    fi
    echo "GPU vendor:  AMD (ROCm)"
    rocm-smi --showid --showtemp --showuse 2>/dev/null || echo "WARNING: rocm-smi not available"

else
    GPU_VENDOR="unknown"
    echo "WARNING: No GPU runtime detected (no nvidia-smi, no /opt/rocm)"
fi

# Resolve enforce-eager: ROCm needs it (torch.compile broken), NVIDIA doesn't.
if [ "$ENFORCE_EAGER" = "auto" ]; then
    if [ "$GPU_VENDOR" = "rocm" ]; then
        ENFORCE_EAGER=true
        echo "Enforce eager: ON (auto — ROCm torch.compile workaround)"
    else
        ENFORCE_EAGER=false
        echo "Enforce eager: OFF (auto — NVIDIA torch.compile works)"
    fi
fi

# ── Python environment ───────────────────────────────────────

OWN_VENV_ROCM="$WORK_BASE/.venvs/llm-services-vllm-rocm"
OWN_VENV_CUDA="$WORK_BASE/.venvs/llm-services-vllm-cuda"
HRABAL_VENV="/lnet/work/home-students-external/hrabal/uv_venv/3.13_vllm0.13_rocm6.4.1"

# Pick the right venv based on GPU vendor
if [ "$GPU_VENDOR" = "nvidia" ]; then
    OWN_VENV="$OWN_VENV_CUDA"
else
    OWN_VENV="$OWN_VENV_ROCM"
fi

if [ "$VLLM_ENV" = "auto" ]; then
    if [ -d "$OWN_VENV" ]; then
        VENV_PATH="$OWN_VENV"
        echo "Using your own venv (auto-detected: $GPU_VENDOR)"
    elif [ -d "$HRABAL_VENV" ]; then
        VENV_PATH="$HRABAL_VENV"
        echo "Using Hrabal's venv (auto-fallback; run setup_env.sh to create your own)"
    else
        echo "ERROR: No venv found."
        echo "       Run setup_env.sh first to create one at $OWN_VENV"
        exit 1
    fi
elif [ "$VLLM_ENV" = "hrabal" ]; then
    VENV_PATH="$HRABAL_VENV"
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

# Activate the venv — for uv-created venvs the activate script may not
# fully work if the Python home isn't on PATH.  We handle this:
#   1. Check pyvenv.cfg for the real Python home
#   2. Prepend it to PATH so the venv's python3 symlink resolves
#   3. Then activate + prepend venv bin as usual

# If pyvenv.cfg specifies a home dir (uv/virtualenv do this), add it to PATH
# so the venv's python3 symlink can resolve.
PYVENV_CFG="$VENV_PATH/pyvenv.cfg"
if [ -f "$PYVENV_CFG" ]; then
    PYHOME=$(grep -E '^home\s*=' "$PYVENV_CFG" | sed 's/^home\s*=\s*//' | tr -d '[:space:]')
    if [ -n "$PYHOME" ] && [ -d "$PYHOME" ]; then
        export PATH="$PYHOME:$PATH"
        echo "Python home (from pyvenv.cfg): $PYHOME"
    fi
fi

source "$VENV_PATH/bin/activate" 2>/dev/null || true
export PATH="$VENV_PATH/bin:$PATH"
echo "Activated venv: $VENV_PATH"

# Verify that python3 actually comes from the venv
ACTIVE_PYTHON="$(which python3 2>/dev/null)"
echo "Python:  $ACTIVE_PYTHON ($(python3 --version 2>&1))"

if [[ "$ACTIVE_PYTHON" != "$VENV_PATH/bin/python3" ]]; then
    # Check if the venv's python3 is a broken symlink
    if [ -L "$VENV_PATH/bin/python3" ] && [ ! -e "$VENV_PATH/bin/python3" ]; then
        LINK_TARGET=$(readlink -f "$VENV_PATH/bin/python3" 2>/dev/null || readlink "$VENV_PATH/bin/python3")
        echo ""
        echo "ERROR: venv's python3 is a broken symlink:"
        echo "       $VENV_PATH/bin/python3 -> $LINK_TARGET"
        echo "       The target Python is not installed on this node."
        echo ""
        echo "  Options:"
        echo "    1. Ask Hrabal where Python 3.13 is installed, add it to PATH"
        echo "    2. Use your own venv: bash setup_env.sh && bash run_vllm.sh --env /lnet/work/people/$USER/.venvs/sprint-vllm-rocm"
        echo ""
        echo "  Diagnostic info:"
        ls -la "$VENV_PATH/bin/python"* 2>/dev/null || true
        cat "$PYVENV_CFG" 2>/dev/null || true
        exit 1
    fi
    echo "WARNING: python3 is not from the venv (expected $VENV_PATH/bin/python3)"
    echo "         Continuing, but vLLM may not be importable."
fi

python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, ROCm: {torch.version.hip}')"

# ── Caches — keep EVERYTHING off $HOME (5 GB quota) ─────────

export HF_HOME="$HF_CACHE"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
mkdir -p "$HF_HOME"
# Disable vLLM usage telemetry (writes to $HOME/.vllm/)
export VLLM_NO_USAGE_STATS=1
# Catch-all: redirect XDG cache base
export XDG_CACHE_HOME="$WORK_BASE/.cache"
mkdir -p "$XDG_CACHE_HOME"

echo "HF cache:      $HF_HOME"
echo "Triton cache:  ${TRITON_CACHE_DIR:-default}"
echo "XDG cache:     $XDG_CACHE_HOME"
echo "vLLM stats:    disabled"

# ── Write connection info ────────────────────────────────────
# This file is read by connect.sh to auto-detect the tunnel target.

CONN_FILE="$WORK_BASE/.sprint_vllm_connection"
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

if [ "$ENFORCE_EAGER" = "true" ]; then
    SERVE_ARGS+=(--enforce-eager)
    echo "Note: --enforce-eager enabled (ROCm torch.compile workaround)"
fi

# Only add data-parallel if > 1 (vLLM may not accept --data-parallel-size 1)
if [ "$DATA_PARALLEL" -gt 1 ]; then
    SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL")
fi

# Enable verbose logging to see worker subprocess errors
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# Increase worker startup timeout for DP>1: 8 workers loading the model
# simultaneously from NFS takes ~2 min (vs ~4s for a single worker).
# Default vLLM timeout (~60s) is too short for this.
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"

exec vllm serve "${SERVE_ARGS[@]}"
