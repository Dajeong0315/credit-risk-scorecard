"""Population Stability Index (PSI) for score/feature stability monitoring.

PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )  over bins,
where `expected` is the reference distribution (e.g. train) and `actual` is the
distribution being monitored (e.g. test, or a later time period).

Standard interpretation: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant shift.
"""
import sqlite3

import numpy as np
import pandas as pd

PSI_MIN_PCT = 1e-4  # floor bin percentages to avoid log(0) / division by zero


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_bins = pd.cut(expected, bins=edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=edges, include_lowest=True)

    expected_pct = (expected_bins.value_counts(sort=False) / len(expected)).clip(lower=PSI_MIN_PCT)
    actual_pct = (actual_bins.value_counts(sort=False) / len(actual)).clip(lower=PSI_MIN_PCT)
    actual_pct = actual_pct.reindex(expected_pct.index).fillna(PSI_MIN_PCT)

    psi = float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())
    return psi


def save_psi(run_id: int, target_name: str, psi_value: float, period: str, conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO psi_monitoring (run_id, target_name, psi_value, period) VALUES (?, ?, ?, ?)",
        (run_id, target_name, psi_value, period),
    )
    conn.commit()
