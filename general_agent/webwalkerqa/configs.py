"""
Experiment configs for GAIA-103 diversity experiment.

9 conditions:
- naive-t4: 4 independent rollouts, no diversity (sequential, k=4).
- jaccard-o{16,32,48,64}: pool o, Jaccard max-min select 4, run 4 threads.
- dense-o{16,32,48,64}: pool o, dense-embedding max-min select 4, run 4 threads.

T=12 turns, n=4 rollouts per question. pass@1 and pass@4.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    """Single experiment configuration."""
    id: str
    method: Literal["sequential", "diversity_parallel"]
    k: int                            # threads/rollouts per question (4)
    n: int                            # turns per rollout (12)
    t: int                            # tokens per turn (unused in this experiment)
    group: str
    o: int = 0                        # pool size for diversity_parallel (16, 32, 48, 64)
    diversity_method: str = "jaccard"  # "jaccard" or "dense"
    description: str = ""

    @property
    def total_tokens(self) -> int:
        return self.k * self.n * self.t

    @property
    def estimated_search_calls(self) -> int:
        return self.k * self.n

    @property
    def summary_tokens(self) -> int:
        return max(128, self.t // 2)


# 9 conditions: naive-t4, jaccard-o16/32/48/64, dense-o16/32/48/64
EXPERIMENT_MATRIX: dict[str, ExperimentConfig] = {
    "naive-t4": ExperimentConfig(
        id="naive-t4",
        method="sequential",
        k=4,
        n=12,
        t=1024,
        group="naive",
        description="4 independent rollouts at temp=1.0, no diversity filtering",
    ),
    "jaccard-o16": ExperimentConfig(
        id="jaccard-o16",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="jaccard",
        o=16,
        diversity_method="jaccard",
        description="Pool 16, Jaccard max-min select 4, 4 threads",
    ),
    "jaccard-o32": ExperimentConfig(
        id="jaccard-o32",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="jaccard",
        o=32,
        diversity_method="jaccard",
        description="Pool 32, Jaccard max-min select 4, 4 threads",
    ),
    "jaccard-o48": ExperimentConfig(
        id="jaccard-o48",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="jaccard",
        o=48,
        diversity_method="jaccard",
        description="Pool 48, Jaccard max-min select 4, 4 threads",
    ),
    "jaccard-o64": ExperimentConfig(
        id="jaccard-o64",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="jaccard",
        o=64,
        diversity_method="jaccard",
        description="Pool 64, Jaccard max-min select 4, 4 threads",
    ),
    "dense-o16": ExperimentConfig(
        id="dense-o16",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="dense",
        o=16,
        diversity_method="dense",
        description="Pool 16, dense MiniLM max-min select 4, 4 threads",
    ),
    "dense-o32": ExperimentConfig(
        id="dense-o32",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="dense",
        o=32,
        diversity_method="dense",
        description="Pool 32, dense MiniLM max-min select 4, 4 threads",
    ),
    "dense-o48": ExperimentConfig(
        id="dense-o48",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="dense",
        o=48,
        diversity_method="dense",
        description="Pool 48, dense MiniLM max-min select 4, 4 threads",
    ),
    "dense-o64": ExperimentConfig(
        id="dense-o64",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="dense",
        o=64,
        diversity_method="dense",
        description="Pool 64, dense MiniLM max-min select 4, 4 threads",
    ),
}


def get_config(config_id: str) -> ExperimentConfig:
    """Get experiment config by ID."""
    if config_id in EXPERIMENT_MATRIX:
        return EXPERIMENT_MATRIX[config_id]
    for key in EXPERIMENT_MATRIX:
        if key.lower() == config_id.lower():
            return EXPERIMENT_MATRIX[key]
    raise ValueError(
        f"Unknown config '{config_id}'. Available: {list(EXPERIMENT_MATRIX.keys())}"
    )


def list_configs(group: str = None) -> list[ExperimentConfig]:
    """List all configs, optionally filtered by group."""
    configs = list(EXPERIMENT_MATRIX.values())
    if group:
        configs = [c for c in configs if c.group.lower() == group.lower()]
    return configs


ALL_CONFIGS = list(EXPERIMENT_MATRIX.values())
