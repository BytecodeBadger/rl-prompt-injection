#!/usr/bin/env bash
# Two-phase curriculum training:
#   Phase 1 — easy bots (DIFFICULTY=easy, 50k steps)  → saves ppo_curriculum_p1
#   Phase 2 — hard bots (DIFFICULTY=hard, 150k steps) → loads ppo_curriculum_p1, saves ppo_redteam
#
# Usage:
#   bash scripts/run_curriculum_training.sh [n_workers]
#   Default n_workers: 8

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_WORKERS=${1:-8}
LOG_DIR="$ROOT_DIR/logs"

[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"

mkdir -p "$LOG_DIR"

stop_bots() {
    echo "Stopping target bots..."
    pkill -f 'uvicorn target_bot:app' 2>/dev/null || true
    sleep 2
}

wait_for_bots() {
    local n=$1
    local base_port=8000
    echo "Waiting for $n bots to be ready..."
    for i in $(seq 0 $((n - 1))); do
        local port=$((base_port + i))
        local attempts=0
        until curl -sf "http://127.0.0.1:$port/docs" >/dev/null 2>&1; do
            attempts=$((attempts + 1))
            if [ $attempts -ge 600 ]; then
                echo "ERROR: Bot on port $port did not start within 600s"
                exit 1
            fi
            sleep 1
        done
    done
    echo "All $n bots ready."
}

# ── Phase 1: easy bots ─────────────────────────────────────────────────────
echo "=== Curriculum Phase 1: easy bots (50k steps) ==="
stop_bots
cd "$ROOT_DIR"
bash scripts/launch_parallel_targets.sh "$N_WORKERS" easy
wait_for_bots "$N_WORKERS"

uv run python main.py --mode curriculum_p1 2>&1 | tee "$LOG_DIR/training_p1.log"

# ── Phase 2: hard bots ─────────────────────────────────────────────────────
echo "=== Curriculum Phase 2: hard bots (150k steps) ==="
stop_bots
bash scripts/launch_parallel_targets.sh "$N_WORKERS" hard
wait_for_bots "$N_WORKERS"

uv run python main.py --mode curriculum_p2 --pretrained-model ppo_curriculum_p1 2>&1 | tee "$LOG_DIR/training_p2.log"

stop_bots
echo "=== Curriculum training complete. Final model: ppo_redteam ==="
