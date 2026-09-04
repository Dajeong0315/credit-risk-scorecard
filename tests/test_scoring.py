import numpy as np
import pandas as pd
import pytest

from src import scoring


def test_probability_to_score_anchor_at_base_odds():
    # odds = (1-p)/p = base_odds  =>  p = 1 / (base_odds + 1)
    base_odds = scoring.DEFAULT_BASE_ODDS
    p_at_base = 1 / (base_odds + 1)
    score = scoring.probability_to_score(np.array([p_at_base]))
    assert score[0] == pytest.approx(scoring.DEFAULT_BASE_SCORE, abs=1e-6)


def test_probability_to_score_doubling_odds_adds_pdo_points():
    base_odds = scoring.DEFAULT_BASE_ODDS
    p_base = 1 / (base_odds + 1)
    p_double = 1 / (2 * base_odds + 1)  # odds = 2 * base_odds
    score_base, score_double = scoring.probability_to_score(np.array([p_base, p_double]))
    assert (score_double - score_base) == pytest.approx(scoring.DEFAULT_PDO, abs=1e-6)


def test_probability_to_score_monotonic_decreasing_in_default_prob():
    p = np.array([0.01, 0.05, 0.1, 0.3, 0.6])
    scores = scoring.probability_to_score(p)
    assert list(scores) == sorted(scores, reverse=True)


def test_assign_grades_five_labels_equal_frequency():
    scores = pd.Series(np.arange(1000))
    grades = scoring.assign_grades(scores)
    counts = grades.value_counts()
    assert set(counts.index) == {"A", "B", "C", "D", "E"}
    assert counts.min() >= 190  # roughly equal-frequency quantiles of 1000/5=200


def test_assign_grades_a_is_highest_score_group():
    scores = pd.Series(np.arange(1000))
    grades = scoring.assign_grades(scores)
    avg_by_grade = pd.DataFrame({"grade": grades, "score": scores}).groupby("grade")["score"].mean()
    assert avg_by_grade["A"] > avg_by_grade["E"]


def test_assign_grades_rejects_non_five():
    with pytest.raises(ValueError):
        scoring.assign_grades(pd.Series(range(10)), n_grades=3)
