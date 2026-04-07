from __future__ import annotations

from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "models"


def ensure_dirs() -> None:
    for path in (DATA_DIR, MODEL_DIR, PLOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clamp(values, low: float, high: float):
    return values.clip(low, high)


def logspace_sample(rng, low: float, high: float, size: int):
    return 10 ** rng.uniform(low, high, size)


def pretty_metric(name: str, value: float) -> str:
    return f"{name}: {value:.4e}" if abs(value) < 1e-2 else f"{name}: {value:.4f}"
