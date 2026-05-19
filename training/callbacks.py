from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from attack_env import ATTACK_PROMPTS, SEED, RedTeamEnv


class MetricsCallback(BaseCallback):
    """Collect per-step and checkpoint metrics during PPO training."""

    def __init__(self, eval_env: RedTeamEnv, eval_every: int, eval_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes

        self.step_rewards: list[float] = []
        self.action_counts = np.zeros(len(ATTACK_PROMPTS), dtype=int)
        self.action_success = np.zeros(len(ATTACK_PROMPTS), dtype=int)

        self.ckpt_timesteps: list[int] = []
        self.ckpt_success_rates: list[float] = []
        self.ckpt_mean_rewards: list[float] = []
        self.ckpt_refusal_rates: list[float] = []
        self.ckpt_api_failure_rates: list[float] = []

        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropy_losses: list[float] = []
        self.clip_fractions: list[float] = []
        self.approx_kls: list[float] = []
        self.explained_variances: list[float] = []

        self.action_entropies: list[float] = []

        self._last_eval_step = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self.step_rewards.append(float(reward))

        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        action_id = info.get("action_id", -1)
        match_type = info.get("match_type", "none")

        if 0 <= action_id < len(ATTACK_PROMPTS):
            self.action_counts[action_id] += 1
            if match_type == "full":
                self.action_success[action_id] += 1

        if hasattr(self.model, "logger") and self.model.logger is not None:
            keys = [
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/clip_fraction",
                "train/approx_kl",
                "train/explained_variance",
            ]
            for key in keys:
                if key in self.model.logger.name_to_value:
                    value = self.model.logger.name_to_value[key]
                    if key == "train/policy_gradient_loss":
                        self.policy_losses.append(float(value))
                    elif key == "train/value_loss":
                        self.value_losses.append(float(value))
                    elif key == "train/entropy_loss":
                        self.entropy_losses.append(float(value))
                    elif key == "train/clip_fraction":
                        self.clip_fractions.append(float(value))
                    elif key == "train/approx_kl":
                        self.approx_kls.append(float(value))
                    elif key == "train/explained_variance":
                        self.explained_variances.append(float(value))

        if self.num_timesteps - self._last_eval_step >= self.eval_every:
            self._run_eval_checkpoint()
            self._last_eval_step = self.num_timesteps

        return True

    def _run_eval_checkpoint(self) -> None:
        successes = 0
        refusals = 0
        api_failures = 0
        ep_rewards: list[float] = []
        action_distributions: list[np.ndarray] = []

        for ep_seed in range(self.eval_episodes):
            obs, _ = self.eval_env.reset(seed=SEED + ep_seed)
            ep_reward = 0.0
            done = False
            episode_had_refusal = False
            episode_had_api_failure = False
            ep_actions = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                ep_actions.append(int(action))
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                ep_reward += float(reward)
                done = terminated or truncated

                if info.get("match_type") == "full":
                    successes += 1
                if info.get("match_type") == "none" and float(self.eval_env._refusal_flag) == 1.0:
                    episode_had_refusal = True
                if info.get("match_type") == "api_error":
                    episode_had_api_failure = True

            refusals += int(episode_had_refusal)
            api_failures += int(episode_had_api_failure)
            ep_rewards.append(ep_reward)

            action_dist = np.bincount(ep_actions, minlength=len(ATTACK_PROMPTS))
            action_distributions.append(action_dist)

        denom = max(1, self.eval_episodes)
        self.ckpt_timesteps.append(self.num_timesteps)
        self.ckpt_success_rates.append(successes / denom)
        self.ckpt_mean_rewards.append(float(np.mean(ep_rewards)))
        self.ckpt_refusal_rates.append(refusals / denom)
        self.ckpt_api_failure_rates.append(api_failures / denom)

        avg_action_dist = np.mean(action_distributions, axis=0)
        if avg_action_dist.sum() > 0:
            action_probs = avg_action_dist / avg_action_dist.sum()
            action_probs = np.clip(action_probs, 1e-8, 1.0)
            entropy = -np.sum(action_probs * np.log(action_probs))
            self.action_entropies.append(float(entropy))

    def to_dict(self) -> dict:
        return {
            "step_rewards": self.step_rewards,
            "action_counts": self.action_counts.tolist(),
            "action_success": self.action_success.tolist(),
            "ckpt_timesteps": self.ckpt_timesteps,
            "ckpt_success_rates": self.ckpt_success_rates,
            "ckpt_mean_rewards": self.ckpt_mean_rewards,
            "ckpt_refusal_rates": self.ckpt_refusal_rates,
            "ckpt_api_failure_rates": self.ckpt_api_failure_rates,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy_losses": self.entropy_losses,
            "clip_fractions": self.clip_fractions,
            "approx_kls": self.approx_kls,
            "explained_variances": self.explained_variances,
            "action_entropies": self.action_entropies,
        }

    def save_metrics(self, path: str | Path) -> None:
        metrics_path = Path(path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
