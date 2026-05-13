# RL Prompt Injection Red Teaming

## Project Overview

This project builds a reinforcement learning workflow for autonomous prompt-injection red teaming. The goal is to train an agent that can systematically probe a defended LLM and discover prompts that bypass safety controls and extract protected data.

## Implemented So Far

Phase 1 (Fortress) is implemented:

- FastAPI target service with one endpoint: `POST /chat`
- Protected assistant prompt that includes an internal session ID loaded from `.env`
- NeMo Guardrails configuration and security flow files
- Runtime SSN-pattern output blocking with safe refusal fallback

Verification and test scaffolding are implemented:

- Lightweight verifier script: `verify_phase1.py`
- Checks API schema, refusal behavior, and SSN-leak blocking
- Pytest coverage for verifier logic plus optional integration check

## Current Status

- Phase 1: complete
- Phase 2 (RL environment and PPO training): not implemented yet

## Quick Commands

- Run service: `uv run uvicorn target_bot:app --host 0.0.0.0 --port 8000`
- Run verifier: `uv run python verify_phase1.py`
- Run tests: `uv run pytest tests/test_phase1_verification.py`

## Interactive Prompt Loop Script

Use the loop client to chat with the Phase 1 server from terminal until cancelled.

- Start loop client: `uv run python scripts/chat_loop.py`
- Custom server URL/timeout: `uv run python scripts/chat_loop.py --base-url http://127.0.0.1:8000 --timeout 30`
- Exit: `Ctrl+C` (or `Ctrl+D`)
