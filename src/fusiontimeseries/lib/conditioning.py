import torch
from contextlib import contextmanager


__all__ = ["ConditionRegistry"]


class ConditionRegistry:
    """A simple store for per-sample parameters."""

    _condition_params: dict[str, torch.Tensor | None] = {}

    @classmethod
    @contextmanager
    def patch(cls, **kwargs: torch.Tensor | None):
        for k, v in kwargs.items():
            cls._condition_params[k] = v
        try:
            yield
        finally:
            cls._condition_params.clear()

    @classmethod
    def get(cls, key: str) -> torch.Tensor | None:
        return cls._condition_params.get(key, None)
