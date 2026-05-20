#!/usr/bin/env bash
set -euo pipefail

# Launches parallel target bots and CLI training in separate tmux sessions.
# Usage:
#   bash scripts/launch_remote_training_tmux.sh [n_workers]
#   Default n_workers: 8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORTRESS_SESSION="fortress"
TRAIN_SESSION="train"
N_WORKERS=${1:-8}
MODE=${2:-normal}   # normal | curriculum

if [[ "$MODE" == "curriculum" ]]; then
  FORTRESS_CMD=""  # curriculum script manages bots internally
  TRAIN_CMD="cd '$ROOT_DIR' && mkdir -p logs && . \$HOME/.local/bin/env && bash scripts/run_curriculum_training.sh $N_WORKERS 2>&1 | tee logs/training.log"
else
  FORTRESS_CMD="cd '$ROOT_DIR' && bash scripts/launch_parallel_targets.sh $N_WORKERS medium && tail -f target_bot_8000.log"
  TRAIN_CMD="cd '$ROOT_DIR' && mkdir -p logs && . \$HOME/.local/bin/env && uv run python main.py --mode normal 2>&1 | tee logs/training.log"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed."
  echo "Install it (Ubuntu): sudo apt-get update && sudo apt-get install -y tmux"
  exit 1
fi

if [[ ! -f "$ROOT_DIR/main.py" ]]; then
  echo "Error: main.py not found at $ROOT_DIR"
  exit 1
fi

if [[ -n "$FORTRESS_CMD" ]]; then
  if tmux has-session -t "$FORTRESS_SESSION" 2>/dev/null; then
    echo "Session '$FORTRESS_SESSION' already exists. Killing and restarting..."
    tmux kill-session -t "$FORTRESS_SESSION"
  fi
  tmux new-session -d -s "$FORTRESS_SESSION" "$FORTRESS_CMD"
  echo "Started tmux session '$FORTRESS_SESSION' ($N_WORKERS target bots)."
fi

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "Session '$TRAIN_SESSION' already exists. Killing and restarting..."
  tmux kill-session -t "$TRAIN_SESSION"
fi
tmux new-session -d -s "$TRAIN_SESSION" "$TRAIN_CMD"
echo "Started tmux session '$TRAIN_SESSION' (training)."

echo
echo "Active tmux sessions:"
tmux ls
echo
echo "Useful commands:"
echo "  Attach API session:      tmux attach -t $FORTRESS_SESSION"
echo "  Attach training session: tmux attach -t $TRAIN_SESSION"
echo "  Watch training details:  tail -f '$ROOT_DIR/logs/training_details.log'"
echo "  Detach from tmux:        Ctrl+b then d"