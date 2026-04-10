#!/bin/bash
# Quick start script for diffusion policy FastAPI server
# Usage: ./start_diffusion_server.sh [port]

set -e

PORT=${1:-9001}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/*/*}"

POLICY_CKPT="/home/rakasheh/master/cassandra/maniskill/checkpoints/diffusion_state/checkpoint_diffusion_policy_stack_cube.pt"
TRAIN_SCRIPT="/home/rakasheh/master/cassandra/maniskill/Maniskill/examples/baselines/diffusion_policy/inference.py"

echo "========================================================================"
echo "Starting Diffusion Policy FastAPI Server"
echo "========================================================================"
echo "Port: $PORT"
echo "Checkpoint: $POLICY_CKPT"
echo "Train script: $TRAIN_SCRIPT"
echo ""
echo "Make sure you're in the diffusion conda environment!"
echo "  conda activate diffusion"
echo ""
echo "========================================================================"

# Check if checkpoint exists
if [ ! -f "$POLICY_CKPT" ]; then
    echo "ERROR: Policy checkpoint not found at:"
    echo "  $POLICY_CKPT"
    exit 1
fi

# Check if train script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Train script not found at:"
    echo "  $TRAIN_SCRIPT"
    exit 1
fi

# Start the server
python "${SCRIPT_DIR}/diffusion_policy_server.py" \
    --policy-ckpt "$POLICY_CKPT" \
    --obs-dim 48 \
    --train-script "$TRAIN_SCRIPT" \
    --host 127.0.0.1 \
    --port "$PORT"

