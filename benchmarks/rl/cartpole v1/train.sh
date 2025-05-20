#!/bin/bash
source .venv/bin/activate

# Create log folder and empty past logs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/bash_logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*

echo "Training..."
python -m benchmarks.rl.cartpole\ v1.cartpole_bash >> "$LOG_DIR/run.log" 2>&1

echo "✅ Training finished!"

