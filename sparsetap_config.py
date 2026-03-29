from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SparseTapConfig:
    seed: int = 42
    max_lag: int = 64
    noise_prob: float = 0.2
    max_taps: int = 16
    beam_width: int = 12
    local_search_restarts: int = 4
    local_search_steps: int = 30
    candidate_top_k: int = 32
    test_prefix: str = "0000010100011010010101100101001110100011110010110011010000111010"
    data_path: Path = field(default_factory=lambda: Path("DAY2_data.txt"))
    artifact_root: Path = field(default_factory=lambda: Path("artifacts"))

    @property
    def candidate_dir(self) -> Path:
        return self.artifact_root / "candidates"

    @property
    def metrics_dir(self) -> Path:
        return self.artifact_root / "metrics"

    @property
    def final_dir(self) -> Path:
        return self.artifact_root / "final"


DEFAULT_CONFIG = SparseTapConfig()


def ensure_artifact_dirs(config: SparseTapConfig = DEFAULT_CONFIG) -> None:
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    config.candidate_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    config.final_dir.mkdir(parents=True, exist_ok=True)
