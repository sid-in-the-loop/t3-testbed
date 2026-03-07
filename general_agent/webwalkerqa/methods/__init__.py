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
from .t3_variants import T3AnchorMethod, T3DiversityJaccardMethod, T3DynamicMethod, T3DynamicJaccardMethod
from .diversity_turn import DiversityTurnMethod

_REGISTRY: dict[str, type] = {
    "s1": S1Method,
    "t3_fixed": T3FixedMethod,
    "t3_anchor": T3AnchorMethod,
    "t3_diversity_jaccard": T3DiversityJaccardMethod,
    "t3_dynamic": T3DynamicMethod,
    "t3_dynamic_jaccard": T3DynamicJaccardMethod,
    "diversity_turn": DiversityTurnMethod,
}


def get_method(name: str) -> type:
    """Return the method class for the given name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


__all__ = [
    "BaseMethod", "MethodResult", "TurnLog",
    "S1Method", "T3FixedMethod", "T3AnchorMethod", "T3DiversityJaccardMethod", "T3DynamicMethod", "T3DynamicJaccardMethod",
    "get_method",
]
