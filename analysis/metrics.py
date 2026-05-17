from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from attack_env import ATTACK_PROMPTS, SEED, RedTeamEnv


def load_training_results(output_dir: str = ".") -> tuple[str, dict, list[dict]]:
    base = Path(output_dir)
    model_path = str(base / "ppo_redteam.zip")
    metrics_path = base / "training_metrics.json"
    hof_path = base / "hall_of_fame.json"

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    with hof_path.open("r", encoding="utf-8") as f:
        hall_of_fame = json.load(f)

    return model_path, metrics, hall_of_fame


def print_training_diagnostics(metrics: dict) -> None:
    rewards = np.array(metrics.get("step_rewards", []), dtype=float)
    action_counts = np.array(metrics.get("action_counts", []), dtype=int)
    action_success = np.array(metrics.get("action_success", []), dtype=int)
    action_entropies = metrics.get("action_entropies", [])

    print("\n=== PPO Training Diagnostics ===")
    print(f"Total training steps: {len(rewards)}")
    print(f"Total episodes (approx): {len(rewards) // 3}")
    print(f"Total reward accumulated: {float(rewards.sum()):.2f}")
    print(f"Mean step reward: {float(rewards.mean()) if len(rewards) else 0.0:.3f}")
    print(f"Std step reward: {float(rewards.std()) if len(rewards) else 0.0:.3f}")

    print("\n--- Action Distribution ---")
    total_actions = int(action_counts.sum())
    for i in range(len(ATTACK_PROMPTS)):
        count = int(action_counts[i]) if i < len(action_counts) else 0
        success = int(action_success[i]) if i < len(action_success) else 0
        pct = 100.0 * count / max(total_actions, 1)
        print(f"Action {i}: {count:4d} uses ({pct:5.1f}%) | {success:3d} successes | {ATTACK_PROMPTS[i][:50]}")

    if action_entropies:
        print("\n--- Exploration Metrics ---")
        print(f"Initial action entropy: {action_entropies[0]:.3f}")
        print(f"Final action entropy: {action_entropies[-1]:.3f}")
        print(f"Max possible entropy: {np.log(len(ATTACK_PROMPTS)):.3f}")


def format_hall_of_fame(hof_list: list[dict]) -> str:
    if not hof_list:
        return "No full leaks detected during training."

    lines = ["| Rank | Action | Hits | First Seen | Prompt |", "|---|---:|---:|---|---|"]
    for rank, entry in enumerate(hof_list, 1):
        first_seen = f"ep {entry.get('first_seen_episode', '?')} step {entry.get('first_seen_step', '?')}"
        prompt = str(entry.get("prompt", "")).replace("|", "\\|")
        lines.append(
            f"| {rank} | {entry.get('action_id', '?')} | {entry.get('hit_count', 0)} | {first_seen} | {prompt} |"
        )
    return "\n".join(lines)


def compute_training_summary(metrics: dict) -> dict:
    action_counts = np.array(metrics.get("action_counts", []), dtype=int)
    action_success = np.array(metrics.get("action_success", []), dtype=int)
    step_rewards = np.array(metrics.get("step_rewards", []), dtype=float)

    return {
        "total_training_steps": int(len(step_rewards)),
        "evaluation_checkpoints": int(len(metrics.get("ckpt_timesteps", []))),
        "best_success_rate": float(max(metrics.get("ckpt_success_rates", [0]))),
        "best_mean_episode_reward": float(max(metrics.get("ckpt_mean_rewards", [0]))),
        "worst_refusal_rate": float(max(metrics.get("ckpt_refusal_rates", [0]))),
        "worst_api_failure_rate": float(max(metrics.get("ckpt_api_failure_rates", [0]))),
        "total_action_uses": int(action_counts.sum()) if len(action_counts) else 0,
        "total_full_leaks": int(action_success.sum()) if len(action_success) else 0,
    }


def run_stochastic_audit(
    model_path: str,
    base_url: str = "http://127.0.0.1:8000",
    n_episodes: int = 30,
    seed: int = SEED,
) -> dict:
    model = PPO.load(model_path)
    audit_env = RedTeamEnv(base_url=base_url, seed=seed + 9999)

    successes = 0
    rewards: list[float] = []

    for ep_seed in range(n_episodes):
        obs, _ = audit_env.reset(seed=seed + ep_seed)
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = audit_env.step(int(action))
            ep_reward += float(reward)
            done = terminated or truncated
            if info.get("match_type") == "full":
                successes += 1
        rewards.append(ep_reward)

    audit_env.close()
    return {
        "success_rate": successes / max(n_episodes, 1),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "episodes": n_episodes,
    }
