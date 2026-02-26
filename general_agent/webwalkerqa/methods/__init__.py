"""
T³ Method implementations.

Available:
  s1       — sequential scaling (single thread, growing token budget)
  t3_fixed — T³ Fixed (k parallel threads, hand-crafted diversity seeds)

Future (not yet implemented, hook points exist):
  t3_dynamic — parent LM decides k and seeds
  t3_dpp     — DPP-based seed selection for maximum diversity

Registry:
  from webwalkerqa.methods import get_method
  method = get_method("t3_fixed")
"""

from .base import BaseMethod, MethodResult, TurnLog
from .s1 import S1Method
from .t3_fixed import T3FixedMethod

_REGISTRY: dict[str, type] = {
    "s1": S1Method,
    "t3_fixed": T3FixedMethod,
    # Future:
    # "t3_dynamic": T3DynamicMethod,
    # "t3_dpp": T3DPPMethod,
}


def get_method(name: str) -> type:
    """Return the method class for the given name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


__all__ = [
    "BaseMethod", "MethodResult", "TurnLog",
    "S1Method", "T3FixedMethod",
    "get_method",
]
