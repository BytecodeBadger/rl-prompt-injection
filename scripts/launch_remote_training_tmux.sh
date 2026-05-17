#!/usr/bin/env bash
set -euo pipefail

# Launches the Phase 1 API and notebook training in separate tmux sessions.
# Usage:
#   bash scripts/launch_remote_training_tmux.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORTRESS_SESSION="fortress"
TRAIN_SESSION="train"

FORTRESS_CMD="cd '$ROOT_DIR' && uv run uvicorn target_bot:app --host 0.0.0.0 --port 8000"
TRAIN_CMD="cd '$ROOT_DIR' && uv run jupyter-nbconvert --to notebook --execute train_agent.ipynb --output train_agent.executed.ipynb --ExecutePreprocessor.timeout=-1 > notebook_execution.log 2>&1"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed."
  echo "Install it (Ubuntu): sudo apt-get update && sudo apt-get install -y tmux"
  exit 1
fi

if [[ ! -f "$ROOT_DIR/train_agent.ipynb" ]]; then
  echo "Error: train_agent.ipynb not found at $ROOT_DIR"
  exit 1
fi

if tmux has-session -t "$FORTRESS_SESSION" 2>/dev/null; then
  echo "Session '$FORTRESS_SESSION' already exists. Leaving it untouched."
else
  tmux new-session -d -s "$FORTRESS_SESSION" "$FORTRESS_CMD"
  echo "Started tmux session '$FORTRESS_SESSION' (target API)."
fi

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "Session '$TRAIN_SESSION' already exists. Killing and restarting..."
  tmux kill-session -t "$TRAIN_SESSION"
fi
tmux new-session -d -s "$TRAIN_SESSION" "$TRAIN_CMD"
echo "Started tmux session '$TRAIN_SESSION' (notebook training)."

echo
echo "Active tmux sessions:"
tmux ls
echo
echo "Useful commands:"
echo "  Attach API session:      tmux attach -t $FORTRESS_SESSION"
echo "  Attach training session: tmux attach -t $TRAIN_SESSION"
echo "  Watch notebook output:   tail -f '$ROOT_DIR/notebook_execution.log'"
echo "  Watch training details:  tail -f '$ROOT_DIR/training_details.log'"
echo "  Detach from tmux:        Ctrl+b then d"