"""Configuration, plus 'freezing' so the held-out test cannot silently use settings edited after the fact."""
from dataclasses import dataclass, asdict
from typing import Optional
import hashlib
import yaml


@dataclass
class SimConfig:
    neutral_threshold_pp: float = 0.5
    rebalance_every_sessions: int = 5
    max_buy_names: Optional[int] = None
    max_position_pct: float = 0.10
    min_sell_fraction: float = 0.25
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_capital: float = 100_000.0
    benchmark_ticker: str = "SPY"
    validation_start: str = "2021-01-01"
    validation_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"

    @property
    def cost_rate(self) -> float:
        """Total one-side cost as a fraction of traded value."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    def validate(self):
        assert self.neutral_threshold_pp >= 0, "threshold must be >= 0"
        assert self.rebalance_every_sessions >= 1
        assert 0 < self.max_position_pct <= 1
        assert 0 < self.min_sell_fraction <= 1
        assert self.initial_capital > 0
        assert self.max_buy_names is None or self.max_buy_names >= 1
        return self


def load_config(path) -> SimConfig:
    with open(path) as f:
        return SimConfig(**yaml.safe_load(f)).validate()


def freeze_config(cfg: SimConfig, path: str):
    """Write the config and a SHA-256 lock file beside it."""
    text = yaml.safe_dump(asdict(cfg), sort_keys=True)
    with open(path, "w") as f:
        f.write(text)
    with open(path + ".sha256", "w") as f:
        f.write(hashlib.sha256(text.encode()).hexdigest())


def verify_frozen(path: str) -> SimConfig:
    """Raise if the frozen config is missing or was edited after freezing."""
    try:
        text = open(path).read()
        expected = open(path + ".sha256").read().strip()
    except FileNotFoundError as e:
        raise RuntimeError(f"Frozen config or lock file missing: {e}")
    if hashlib.sha256(text.encode()).hexdigest() != expected:
        raise RuntimeError("Frozen config was modified after freezing; refusing to run the test.")
    return SimConfig(**yaml.safe_load(text)).validate()
