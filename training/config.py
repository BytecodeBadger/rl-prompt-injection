from __future__ import annotations

import logging
from pathlib import Path

from attack_env import SEED


def get_training_config(mode: str, output_dir: str = ".") -> dict:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"quick", "normal"}:
        raise ValueError("mode must be 'quick' or 'normal'")

    quick_pass = normalized_mode == "quick"
    out_dir = Path(output_dir)

    if quick_pass:
        total_timesteps = 50
        eval_every = 25
        eval_episodes = 5
        audit_episodes = 5
    else:
        total_timesteps = 20_000
        eval_every = 500
        eval_episodes = 30
        audit_episodes = 30

    return {
        "mode": normalized_mode,
        "quick_pass": quick_pass,
        "seed": SEED,
        "base_url": "http://127.0.0.1:8000",
        "base_port": 8000,
        "n_workers": 8,
        "total_timesteps": total_timesteps,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
        "audit_episodes": audit_episodes,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "gamma": 0.99,
        "ent_coef": 0.1,
        "device": "cpu",
        "output_dir": str(out_dir),
        "model_path": str(out_dir / "ppo_redteam"),
        "metrics_path": str(out_dir / "training_metrics.json"),
        "hall_of_fame_path": str(out_dir / "hall_of_fame.json"),
        "logs_dir": str(out_dir / "logs"),
        "training_log_path": str(out_dir / "logs" / "training_details.log"),
    }


def configure_logging(config: dict) -> None:
    logs_dir = Path(config["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    attack_logger = logging.getLogger("attack_env")
    sb3_logger = logging.getLogger("stable_baselines3")

    attack_logger.handlers.clear()
    sb3_logger.handlers.clear()

    if config["quick_pass"]:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        attack_logger.addHandler(console_handler)
        attack_logger.setLevel(logging.DEBUG)
        sb3_logger.setLevel(logging.INFO)
    else:
        file_handler = logging.FileHandler(config["training_log_path"], mode="w")
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        console_handler.setLevel(logging.WARNING)

        attack_logger.addHandler(file_handler)
        attack_logger.addHandler(console_handler)
        attack_logger.setLevel(logging.DEBUG)
        sb3_logger.setLevel(logging.WARNING)
