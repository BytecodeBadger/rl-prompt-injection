#!/bin/bash
# Launch multiple target bot instances for parallel RL training

set -e

# Ensure uv is on PATH (installed to ~/.local/bin on EC2)
# shellcheck source=/dev/null
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"

# Number of parallel workers
N_WORKERS=${1:-4}
BASE_PORT=8000
DIFFICULTY=${2:-hard}

echo "Launching $N_WORKERS target bot instances (difficulty=$DIFFICULTY)..."

# Kill existing instances on these ports (if any)
for i in $(seq 0 $((N_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Killing existing process on port $PORT (PID: $PID)"
        kill -9 $PID 2>/dev/null || true
        sleep 0.5
    fi
done

# Launch new instances
for i in $(seq 0 $((N_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "Starting target bot on port $PORT..."
    DIFFICULTY=$DIFFICULTY uv run uvicorn target_bot:app --host 0.0.0.0 --port $PORT > "target_bot_$PORT.log" 2>&1 &
    sleep 1
done

echo ""
echo "✓ Launched $N_WORKERS target bot instances:"
for i in $(seq 0 $((N_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "  - http://127.0.0.1:$PORT (log: target_bot_$PORT.log)"
done

echo ""
echo "To stop all instances:"
echo "  pkill -f 'uvicorn target_bot:app'"
