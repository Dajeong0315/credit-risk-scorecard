"""Probability-to-score scaling (PDO method) and A-E grade assignment."""
import sqlite3

import numpy as np
import pandas as pd

DEFAULT_BASE_SCORE = 600
DEFAULT_BASE_ODDS = 50  # good:bad odds at the base score
DEFAULT_PDO = 20        # points to double the odds

GRADE_LABELS_ASCENDING = ["E", "D", "C", "B", "A"]  # E = highest risk, A = lowest risk


def probability_to_score(p_default: np.ndarray, base_score: int = DEFAULT_BASE_SCORE,
                          base_odds: float = DEFAULT_BASE_ODDS, pdo: float = DEFAULT_PDO) -> np.ndarray:
    """Standard scorecard PDO scaling.

    odds = P(good) / P(bad); score = offset + factor * ln(odds)
    factor = pdo / ln(2); offset = base_score - factor * ln(base_odds)
    => at odds == base_odds, score == base_score; doubling odds adds `pdo` points.
    """
    p = np.clip(np.asarray(p_default, dtype=float), 1e-6, 1 - 1e-6)
    odds = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds)


def assign_grades(scores: pd.Series, n_grades: int = 5) -> pd.Series:
    """Equal-frequency (quantile) grading. A = best (highest score / lowest risk)."""
    if n_grades != 5:
        raise ValueError("이 프로젝트는 A-E 5등급 체계를 사용합니다 (n_grades=5)")
    ranks = scores.rank(method="first")
    grades = pd.qcut(ranks, q=n_grades, labels=GRADE_LABELS_ASCENDING)
    return grades.astype(str)


def save_scores(ids: np.ndarray, scores: np.ndarray, grades: pd.Series, run_id: int,
                 conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM scores WHERE run_id = ?", (run_id,))
    rows = list(zip([run_id] * len(ids), [int(i) for i in ids], [float(s) for s in scores], list(grades)))
    conn.executemany("INSERT INTO scores (run_id, sk_id_curr, score, grade) VALUES (?, ?, ?, ?)", rows)
    conn.commit()
