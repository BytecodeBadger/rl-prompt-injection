from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from attack_env import ATTACK_PROMPTS


def _load_metrics(metrics_path: str) -> dict:
    with Path(metrics_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_metrics(metrics_path: str, save_path: str | None = None):
    metrics = _load_metrics(metrics_path)
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PPO Red-Team Training Metrics", fontsize=14)

    rewards = np.array(metrics.get("step_rewards", []), dtype=float)
    cumulative = np.cumsum(rewards) if len(rewards) else np.array([])
    sns.lineplot(x=np.arange(len(cumulative)), y=cumulative, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_xlabel("Training Steps")
    axes[0, 0].set_ylabel("Cumulative Reward")
    axes[0, 0].set_title("Reward Trend")

    sns.lineplot(
        x=metrics.get("ckpt_timesteps", []),
        y=metrics.get("ckpt_success_rates", []),
        ax=axes[0, 1],
        marker="o",
        color="green",
    )
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Full Leak Success Rate")
    axes[0, 1].set_title("Success Rate @ Checkpoints")
    axes[0, 1].set_ylim(-0.05, 1.05)

    sns.lineplot(
        x=metrics.get("ckpt_timesteps", []),
        y=metrics.get("ckpt_mean_rewards", []),
        ax=axes[1, 0],
        marker="o",
        color="orange",
    )
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Mean Episode Reward")
    axes[1, 0].set_title("Mean Episode Reward @ Checkpoints")

    sns.lineplot(
        x=metrics.get("ckpt_timesteps", []),
        y=metrics.get("ckpt_refusal_rates", []),
        ax=axes[1, 1],
        marker="o",
        color="red",
        label="Refusal Episode Rate",
    )
    sns.lineplot(
        x=metrics.get("ckpt_timesteps", []),
        y=metrics.get("ckpt_api_failure_rates", []),
        ax=axes[1, 1],
        marker="s",
        color="purple",
        label="API Failure Episode Rate",
    )
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_title("Refusal & API Failure Episode Rates")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_ppo_diagnostics(metrics_path: str, save_path: str | None = None):
    metrics = _load_metrics(metrics_path)
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    action_entropies = metrics.get("action_entropies", [])
    ckpt_timesteps = metrics.get("ckpt_timesteps", [])

    if action_entropies:
        max_entropy = np.log(len(ATTACK_PROMPTS))
        sns.lineplot(
            x=ckpt_timesteps[: len(action_entropies)],
            y=action_entropies,
            ax=axes[0],
            marker="o",
            color="purple",
            label="Actual Entropy",
        )
        axes[0].axhline(y=max_entropy, color="red", linestyle="--", alpha=0.5, label=f"Max Entropy ({max_entropy:.2f})")
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Action Entropy (nats)")
        axes[0].set_title("Action Exploration Entropy Over Training")
        axes[0].legend()
        axes[0].set_ylim(-0.1, max_entropy + 0.2)
    else:
        axes[0].text(0.5, 0.5, "No entropy data collected", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Action Exploration Entropy")

    explained_variances = metrics.get("explained_variances", [])
    if explained_variances:
        sns.lineplot(
            x=range(len(explained_variances)),
            y=explained_variances,
            ax=axes[1],
            marker="o",
            color="green",
            label="Explained Variance",
        )
        axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Zero Baseline")
        axes[1].axhline(y=1, color="blue", linestyle="--", alpha=0.5, label="Perfect")
        axes[1].set_xlabel("Policy Update")
        axes[1].set_ylabel("Explained Variance")
        axes[1].set_title("Value Function Learning Quality")
        axes[1].legend()
        axes[1].set_ylim(-0.1, 1.1)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No PPO metrics captured",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("PPO Value Function Quality")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_attack_success(metrics_path: str, save_path: str | None = None):
    metrics = _load_metrics(metrics_path)
    action_success = metrics.get("action_success", [0] * len(ATTACK_PROMPTS))
    sns.set_theme(style="darkgrid")

    short_labels = [f"Action {i}" for i in range(len(ATTACK_PROMPTS))]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=short_labels, y=action_success, ax=ax, palette="muted")
    ax.set_xlabel("Attack Shorthand")
    ax.set_ylabel("Full Leak Count")
    ax.set_title("Attack Shorthand Success Counts (Full Leaks Only)")

    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    legend_text = "\n".join([f"Action {i}: {ATTACK_PROMPTS[i][:70]}" for i in range(len(ATTACK_PROMPTS))])
    fig.text(
        0.01,
        -0.35,
        legend_text,
        fontsize=7,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
