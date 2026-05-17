from __future__ import annotations

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from attack_env import RedTeamEnv


def make_env(port: int, rank: int, seed: int):
    """Environment factory for SubprocVecEnv workers."""

    def _init():
        return RedTeamEnv(base_url=f"http://127.0.0.1:{port}", seed=seed + rank)

    return _init


def create_vectorized_env(config: dict):
    workers = config["n_workers"]
    base_port = config["base_port"]
    seed = config["seed"]
    train_env = SubprocVecEnv([make_env(base_port + i, i, seed) for i in range(workers)])
    return VecMonitor(train_env)


def create_eval_env(base_url: str, seed: int) -> RedTeamEnv:
    return RedTeamEnv(base_url=base_url, seed=seed)
