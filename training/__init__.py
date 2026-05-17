def run_training(*args, **kwargs):
	from .train import run_training as _run_training

	return _run_training(*args, **kwargs)


__all__ = ["run_training"]
