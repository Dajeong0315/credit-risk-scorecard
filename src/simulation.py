"""Cutoff simulation: approval rate / expected default rate / expected loss by score cutoff."""
import sqlite3

import numpy as np
import pandas as pd

# Loss-given-default assumption used to translate a default rate into an expected
# loss rate (fraction of exposure lost). 45% is the Basel II/III standard
# unsecured-retail LGD assumption; documented here since it is a modeling choice.
DEFAULT_LGD = 0.45


def simulate_cutoffs(scores: pd.Series, targets: pd.Series, percentiles: np.ndarray = None,
                      lgd: float = DEFAULT_LGD) -> pd.DataFrame:
    if percentiles is None:
        percentiles = np.arange(5, 100, 5)
    cutoffs = np.percentile(scores, percentiles)

    rows = []
    for cutoff in cutoffs:
        approved = scores >= cutoff
        approval_rate = float(approved.mean())
        expected_default_rate = float(targets[approved].mean()) if approved.sum() > 0 else 0.0
        expected_loss = expected_default_rate * lgd
        rows.append({
            "cutoff": float(cutoff),
            "approval_rate": approval_rate,
            "expected_default_rate": expected_default_rate,
            "expected_loss": expected_loss,
        })
    return pd.DataFrame(rows).drop_duplicates(subset="cutoff").sort_values("cutoff").reset_index(drop=True)


def save_cutoff_simulation(sim_df: pd.DataFrame, run_id: int, conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM cutoff_simulation WHERE run_id = ?", (run_id,))
    rows = [
        (run_id, r.cutoff, r.approval_rate, r.expected_default_rate, r.expected_loss)
        for r in sim_df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT INTO cutoff_simulation (run_id, cutoff, approval_rate, expected_default_rate, expected_loss) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
