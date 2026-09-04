"""Feature drift monitoring for detecting distribution shifts in production data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_psi(
    reference: pd.Series, current: pd.Series, n_bins: int = 10, epsilon: float = 1e-10
) -> float:
    """Compute Population Stability Index (PSI) to measure feature drift."""

    def _psi_for_series(ref: pd.Series, cur: pd.Series) -> float:
        min_val = min(ref.min(), cur.min())
        max_val = max(ref.max(), cur.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        ref_counts = np.histogram(ref, bins=bins)[0] + epsilon
        cur_counts = np.histogram(cur, bins=bins)[0] + epsilon

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi_value)

    return _psi_for_series(reference, current)


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str] | None = None,
    psi_threshold: float = 0.1,
) -> dict[str, Any]:
    """Detect feature drift across multiple features via PSI scores."""
    if columns is None:
        columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    psi_scores: dict[str, float] = {}
    for col in columns:
        if col in reference_df.columns and col in current_df.columns:
            psi_scores[col] = compute_psi(reference_df[col], current_df[col])

    has_drift = any(psi > psi_threshold for psi in psi_scores.values())

    return {"has_drift": has_drift, "psi_by_feature": psi_scores}
