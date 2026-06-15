from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    method: Literal["sequential", "diversity_parallel"]
    k: int
    n: int
    t: int
    group: str
    o: int = 0
    diversity_method: str = "jaccard"
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
    "jaccard-o8": ExperimentConfig(
        id="jaccard-o8",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="jaccard",
        o=8,
        diversity_method="jaccard",
        description="Pool 8, Jaccard max-min select 4, 4 threads",
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
    "dense-o8": ExperimentConfig(
        id="dense-o8",
        method="diversity_parallel",
        k=4,
        n=12,
        t=1024,
        group="dense",
        o=8,
        diversity_method="dense",
        description="Pool 8, dense MiniLM max-min select 4, 4 threads",
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
