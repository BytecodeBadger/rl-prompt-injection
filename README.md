# RL Prompt Injection Red Teaming

## Project Overview

This project builds a reinforcement learning workflow for autonomous prompt-injection red teaming. The goal is to train an agent that can systematically probe a defended LLM and discover prompts that bypass safety controls and extract protected data.

### Phase 1: Guarded Target Service (implemented)

- FastAPI target service with one endpoint: `POST /chat` (`target_bot.py`)
- Protected system prompt with session ID loaded from `.env` (`SESSION_ID`)
- Guardrail artifacts in `guardrails/config.yml` and `guardrails/security.co`
- Runtime SSN-pattern output blocking with safe refusal fallback
- Verifier script (`verify_phase1.py`) for schema/refusal/no-leak checks
- Pytest coverage for verifier logic and optional integration verification

### Phase 2: RL Environment + Training Pipeline (implemented)

- Gymnasium environment in `attack_env.py` with:
	- Discrete action space of 6 prompt-injection shorthand attacks
	- Reward shaping for full leak, partial leak, refusals, and exploration
	- Multi-turn episodes (`MAX_STEPS = 3`)
	- Retry/error handling for API calls and early stop on repeated failures
	- Hall of Fame tracking and JSON persistence (`hall_of_fame.json`)
- Python training pipeline in `training/`:
	- `training/train.py`: CLI and orchestration entrypoint (`run_training`)
	- `training/config.py`: quick/normal mode config and logging setup
	- `training/callbacks.py`: PPO metrics collection and JSON persistence
	- `training/env_factory.py`: vectorized env and eval env creation
	- `training/preflight.py`: service/security/environment validation checks
- Analysis package in `analysis/`:
	- `analysis/visualization.py`: plot generation from saved metrics artifacts
	- `analysis/metrics.py`: summaries, Hall of Fame formatting, stochastic audit
- Notebook assets:
	- `train_agent.ipynb`: post-training analysis and visualization only
	- `train_agent.executed.ipynb`: optional executed artifact
- Training throughput is improved through parallelization (multiple environment workers)
- Training/runtime logs directory (`logs/`) with API and training logs

## Quick Start

### 1) Start target bots for parallel training (recommended)

Training uses parallel environment workers by default (`n_workers = 8`), so launch one target API per worker port (`8000-8007`):

```bash
bash scripts/launch_parallel_targets.sh 8
```

Single-instance API mode (useful for quick manual checks only):

```bash
uv run uvicorn target_bot:app --host 0.0.0.0 --port 8000
```

### 2) Verify Phase 1 behavior

```bash
uv run python verify_phase1.py
```

### 3) Run tests

```bash
uv run pytest
```

Run only verifier tests:

```bash
uv run pytest tests/test_phase1_verification.py
```

Run integration test (requires running API):

```bash
RUN_PHASE1_VERIFY=1 uv run pytest tests/test_phase1_verification.py -m integration
```

### 4) Run RL training (from Python/CLI, not notebook)

Before training, make sure target APIs are running on matching worker ports (default: 8 workers => ports `8000-8007`).

Quick verification run:

```bash
uv run python main.py --mode quick
```

Full training run:

```bash
uv run python main.py --mode normal
```

Equivalent script entrypoint:

```bash
uv run rl-train --mode normal
```

Optional output directory:

```bash
uv run python main.py --mode normal --output-dir ./artifacts
```

### 5) Run notebook for post-training analysis only

After training artifacts are available (`ppo_redteam.zip`, `training_metrics.json`, `hall_of_fame.json`), open and run `train_agent.ipynb`.

## Useful Scripts

Interactive chat loop against the target API:

```bash
uv run python scripts/chat_loop.py
```

Custom server URL/timeout:

```bash
uv run python scripts/chat_loop.py --base-url http://127.0.0.1:8000 --timeout 30
```

Launch multiple target API workers (training default: 8):

```bash
bash scripts/launch_parallel_targets.sh 8
```

Launch API + training workflows in tmux sessions:

```bash
bash scripts/launch_remote_training_tmux.sh
```

## Repository Layout

- `target_bot.py`: Phase 1 guarded target API
- `verify_phase1.py`: verification checks for Phase 1
- `attack_env.py`: Phase 2 RL environment and reward logic
- `training/`: RL training orchestration package (CLI + helpers)
- `analysis/`: post-training analysis and visualization package
- `train_agent.ipynb`: post-training analysis notebook (no training orchestration)
- `scripts/`: helper launch and client scripts
- `guardrails/`: minimal NeMo guardrail configuration
- `tests/`: verification-focused test suite
- `docs/`: implementation and workflow notes

## Notes

- Use `uv` for dependency management and command execution.
- Keep secrets in `.env` (do not commit `.env`; commit `.env.example` if needed).
- `main.py` is the training CLI wrapper entrypoint.
- Training should be initiated from Python entrypoints (`main.py` or `rl-train`), not from notebooks.
