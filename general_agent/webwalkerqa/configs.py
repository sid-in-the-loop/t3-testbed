"""
Experiment matrix for WebWalkerQA T³ PoC.

Each config specifies method, k (threads/search-scale), n (turns), t (tokens/thread/turn).
s1: 1 thread, t tokens per turn, model may use multiple searches within the budget.
t3_fixed: k parallel threads, each gets t tokens + 1 search, parent synthesizes.
oracle: pass@8 — 8 independent s1 runs, oracle selects best answer per question.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    """Single experiment configuration."""
    id: str                           # e.g. "A1", "B2"
    method: Literal["s1", "t3_fixed", "oracle"]
    k: int                            # threads (T³) or search-scale factor (s1)
    n: int                            # turns
    t: int                            # tokens per thread per turn
    group: str                        # compute-matched group: "A", "B", "C"
    description: str = ""

    @property
    def total_tokens(self) -> int:
        return self.k * self.n * self.t

    @property
    def estimated_search_calls(self) -> int:
        """Estimated search calls (T³: k*n; s1: n*floor(t/1024))."""
        if self.method == "t3_fixed":
            return self.k * self.n
        elif self.method == "oracle":
            return 8 * self.n  # 8 independent runs
        else:  # s1
            searches_per_turn = max(1, self.t // 1024)
            return self.n * searches_per_turn

    @property
    def summary_tokens(self) -> int:
        """Summary token budget per thread (s = t/2)."""
        return max(128, self.t // 2)


# Full experiment matrix
EXPERIMENT_MATRIX: dict[str, ExperimentConfig] = {
    # ── Group A: 6,144 total tokens ──────────────────────────────────────────
    "A1": ExperimentConfig(
        id="A1", method="s1", k=1, n=6, t=1024,
        group="A", description="s1 baseline: 1 thread, 1024 tokens/turn",
    ),
    "A2": ExperimentConfig(
        id="A2", method="t3_fixed", k=2, n=6, t=512,
        group="A", description="T³ Fixed: 2 threads, 512 tokens/thread/turn",
    ),

    # ── Group B: 24,576 total tokens ─────────────────────────────────────────
    "B1": ExperimentConfig(
        id="B1", method="s1", k=1, n=6, t=4096,
        group="B", description="s1 baseline: 1 thread, 4096 tokens/turn",
    ),
    "B2": ExperimentConfig(
        id="B2", method="t3_fixed", k=4, n=6, t=1024,
        group="B", description="T³ Fixed: 4 threads, 1024 tokens/thread/turn",
    ),
    "B3": ExperimentConfig(
        id="B3", method="t3_fixed", k=8, n=6, t=512,
        group="B", description="T³ Fixed: 8 threads, 512 tokens/thread/turn",
    ),

    # ── Group C: 49,152 total tokens ─────────────────────────────────────────
    "C1": ExperimentConfig(
        id="C1", method="s1", k=1, n=6, t=8192,
        group="C", description="s1 baseline: 1 thread, 8192 tokens/turn",
    ),
    "C2": ExperimentConfig(
        id="C2", method="t3_fixed", k=8, n=6, t=1024,
        group="C", description="T³ Fixed: 8 threads, 1024 tokens/thread/turn",
    ),
    "C3": ExperimentConfig(
        id="C3", method="t3_fixed", k=16, n=6, t=512,
        group="C", description="T³ Fixed: 16 threads, 512 tokens/thread/turn",
    ),

    # ── Oracle: uncapped ceiling ──────────────────────────────────────────────
    "Oracle": ExperimentConfig(
        id="Oracle", method="oracle", k=8, n=6, t=1024,
        group="Oracle", description="Oracle: pass@8 (best of 8 independent s1 runs)",
    ),
}


def get_config(config_id: str) -> ExperimentConfig:
    """Get experiment config by ID (case-insensitive)."""
    key = config_id.upper()
    if key not in EXPERIMENT_MATRIX:
        raise ValueError(
            f"Unknown config '{config_id}'. Available: {list(EXPERIMENT_MATRIX.keys())}"
        )
    return EXPERIMENT_MATRIX[key]


def list_configs(group: str = None) -> list[ExperimentConfig]:
    """List all configs, optionally filtered by group."""
    configs = list(EXPERIMENT_MATRIX.values())
    if group:
        configs = [c for c in configs if c.group.upper() == group.upper()]
    return configs


# Grouped for easy iteration
S1_CONFIGS = [c for c in EXPERIMENT_MATRIX.values() if c.method == "s1"]
T3_CONFIGS = [c for c in EXPERIMENT_MATRIX.values() if c.method == "t3_fixed"]
ALL_CONFIGS = list(EXPERIMENT_MATRIX.values())
