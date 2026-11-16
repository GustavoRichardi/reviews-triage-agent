"""Estatísticas pequenas usadas na validação do juiz e na revisão dos rótulos."""

import math
import warnings

import numpy as np
from sklearn.metrics import cohen_kappa_score


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """IC de 95% para uma proporção; funciona bem mesmo com n pequeno ou p perto de 0/1."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def kappa(a, b, labels: list, weights: str | None = None) -> float | None:
    """Kappa de Cohen: concordância além do acaso. Com `weights="quadratic"` pune mais as
    discordâncias grandes (para escalas ordenadas). None quando indefinido (sem variação)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = cohen_kappa_score(a, b, labels=labels, weights=weights)
    return None if np.isnan(value) else float(value)
