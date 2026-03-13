"""
Method implementations for GAIA-103 diversity experiment.

- sequential: Naive parallel (k independent ReAct rollouts, no diversity).
- diversity_parallel: One pool of o candidates, max-min select k seeds, run k threads.
"""

from .base import BaseMethod, MethodResult, TurnLog
from .diversity_scaling import SequentialMethod, DiversityParallelMethod

_REGISTRY: dict[str, type] = {
    "sequential": SequentialMethod,
    "diversity_parallel": DiversityParallelMethod,
}


def get_method(name: str) -> type:
    """Return the method class for the given name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


__all__ = [
    "BaseMethod",
    "MethodResult",
    "TurnLog",
    "SequentialMethod",
    "DiversityParallelMethod",
    "get_method",
]
