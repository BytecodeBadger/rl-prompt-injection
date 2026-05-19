from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from stable_baselines3 import PPO

from analysis.metrics import format_text_analysis
from attack_env import save_hall_of_fame
from training.callbacks import MetricsCallback
from training.config import configure_logging, get_training_config
from training.env_factory import create_eval_env, create_vectorized_env
from training.notify import notify_training_analysis, notify_training_complete, notify_training_failed, notify_training_started
from training.preflight import (
    check_service_connectivity,
    validate_environment,
    validate_fortress_security,
)

log = logging.getLogger(__name__)


def run_training(mode: str = "normal", output_dir: str = ".") -> dict:
    config = get_training_config(mode=mode, output_dir=output_dir)
    configure_logging(config)

    check_service_connectivity(config["base_url"])
    validate_fortress_security(config["base_url"])
    env_for_check = create_eval_env(config["base_url"], config["seed"])
    validate_environment(env_for_check)

    train_env = create_vectorized_env(config)
    eval_env = create_eval_env(config["base_url"], config["seed"] + 1000)

    callback = MetricsCallback(
        eval_env=eval_env,
        eval_every=config["eval_every"],
        eval_episodes=config["eval_episodes"],
    )

    n_workers = config["n_workers"]
    n_steps_per_env = 128 // n_workers if n_workers > 1 else 128
    n_steps_per_env = max(32, n_steps_per_env)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config["learning_rate"],
        n_steps=n_steps_per_env,
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        seed=config["seed"],
        device=config["device"],
        verbose=1,
    )

    model.learn(total_timesteps=config["total_timesteps"], callback=callback)

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    model.save(config["model_path"])
    callback.save_metrics(config["metrics_path"])
    save_hall_of_fame(config["hall_of_fame_path"])

    train_env.close()
    eval_env.close()
    env_for_check.close()
    return callback.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO training for RL prompt-injection agent")
    parser.add_argument("--mode", choices=["quick", "normal"], default="normal")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    config = get_training_config(mode=args.mode, output_dir=args.output_dir)
    notify_training_started(args.mode)
    try:
        metrics = run_training(mode=args.mode, output_dir=args.output_dir)
        notify_training_complete(args.mode, metrics or {})

        hof_path = Path(config["hall_of_fame_path"])
        hall_of_fame = json.loads(hof_path.read_text(encoding="utf-8")) if hof_path.exists() else []
        analysis = format_text_analysis(metrics or {}, hall_of_fame)

        log_path = Path(config["training_log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n{analysis}\n")

        notify_training_analysis(analysis)
    except Exception as exc:
        notify_training_failed(args.mode, exc)
        raise


if __name__ == "__main__":
    main()
