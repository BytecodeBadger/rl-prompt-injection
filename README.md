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

### Phase 2: RL Environment + Training Assets (implemented)

- Gymnasium environment in `attack_env.py` with:
	- Discrete action space of 6 prompt-injection shorthand attacks
	- Reward shaping for full leak, partial leak, refusals, and exploration
	- Multi-turn episodes (`MAX_STEPS = 3`)
	- Retry/error handling for API calls and early stop on repeated failures
	- Hall of Fame tracking and JSON persistence (`hall_of_fame.json`)
- Training notebook assets:
	- `train_agent.ipynb`
	- `train_agent.executed.ipynb` (executed output artifact)
- Training throughput is improved through parallelization (running multiple target API workers during training workflows)
- Training/runtime logs directory (`logs/`) with API and training logs

## Quick Start

### 1) Start the target bot

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

## Useful Scripts

Interactive chat loop against the target API:

```bash
uv run python scripts/chat_loop.py
```

Custom server URL/timeout:

```bash
uv run python scripts/chat_loop.py --base-url http://127.0.0.1:8000 --timeout 30
```

Launch multiple target API workers (default 4):

```bash
bash scripts/launch_parallel_targets.sh 4
```

Launch API + notebook training in tmux sessions:

```bash
bash scripts/launch_remote_training_tmux.sh
```

## Repository Layout

- `target_bot.py`: Phase 1 guarded target API
- `verify_phase1.py`: verification checks for Phase 1
- `attack_env.py`: Phase 2 RL environment and reward logic
- `train_agent.ipynb`: PPO training/evaluation notebook
- `scripts/`: helper launch and client scripts
- `guardrails/`: minimal NeMo guardrail configuration
- `tests/`: verification-focused test suite
- `docs/`: implementation and workflow notes

## Notes

- Use `uv` for dependency management and command execution.
- Keep secrets in `.env` (do not commit `.env`; commit `.env.example` if needed).
- `main.py` is currently a placeholder entrypoint and not part of the main training/serving flow.
