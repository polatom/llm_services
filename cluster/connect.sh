#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SSH tunnel to reach the SPRINT vLLM server from your local machine.
#
# WHY: The GPU node running vLLM is on the cluster's internal network.
# It's not reachable directly from outside UFAL. An SSH tunnel forwards
# a port on your local machine through the cluster login node (lrc1)
# to the GPU node, so your SPRINT backend can reach the vLLM API at
# http://localhost:8421/v1/chat/completions.
#
# Usage:
#   bash connect.sh                        # auto-detect from connection file
#   bash connect.sh tdll-8gpu5 8421        # manual: hostname port
#
# The connection file (~/.sprint_vllm_connection) is written automatically
# by run_vllm.sh when the vLLM server starts.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SUBMIT_NODE="${LRC_SUBMIT_NODE:-lrc1.ufal.hide.ms.mff.cuni.cz}"
CONN_FILE="$HOME/.sprint_vllm_connection"

# ── Determine host and port ──────────────────────────────────

VLLM_MODEL="(unknown)"

if [ $# -ge 2 ]; then
    VLLM_HOST="$1"
    VLLM_PORT="$2"
elif [ -f "$CONN_FILE" ]; then
    source "$CONN_FILE"
    echo "Read connection info from $CONN_FILE"
else
    echo "Usage: $0 [hostname port]"
    echo ""
    echo "Or start vLLM first (bash run_vllm.sh on the GPU node) — it"
    echo "writes connection info to $CONN_FILE automatically."
    echo ""
    echo "To find your GPU node, check your Slurm job:"
    echo "  ssh $SUBMIT_NODE squeue -u \$USER"
    exit 1
fi

LOCAL_PORT="${LOCAL_PORT:-$VLLM_PORT}"

echo "======================================================"
echo "  SPRINT vLLM Tunnel"
echo "  Remote:  $VLLM_HOST:$VLLM_PORT (GPU node on cluster)"
echo "  Local:   localhost:$LOCAL_PORT (your machine)"
echo "  Via:     $SUBMIT_NODE (SSH jump host)"
echo "  Model:   $VLLM_MODEL"
echo "======================================================"
echo ""
echo "Once the tunnel is open, test with:"
echo "  curl -s http://localhost:$LOCAL_PORT/v1/models | python3 -m json.tool"
echo ""
echo "Add to your SPRINT .env:"
echo "  LLM_PROVIDER=UFAL_TP"
echo "  UFAL_TP_ENDPOINT=http://localhost:$LOCAL_PORT/v1/chat/completions"
echo "  UFAL_TP_MODEL=$VLLM_MODEL"
echo "  UFAL_TP_APIKEY=dummy"
echo "  LLM_MAX_CONCURRENCY=48"
echo "  LLM_BATCH_SIZE=15"
echo ""
echo "Opening SSH tunnel (Ctrl+C to close)..."
echo "  ssh -N -L $LOCAL_PORT:$VLLM_HOST:$VLLM_PORT $SUBMIT_NODE"
echo ""

ssh -N -L "$LOCAL_PORT:$VLLM_HOST:$VLLM_PORT" "$SUBMIT_NODE"
