"""Baseline scorecard: logistic regression on WoE-transformed, IV-selected features."""
import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import woe_binning as wb

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    return float(ks_2samp(pos, neg).statistic)


def gini_from_auc(auc: float) -> float:
    return float(2 * auc - 1)


def fit_woe_for_all_features(train_df: pd.DataFrame, feature_cols: list, target_col: str = TARGET_COL) -> tuple:
    fits = {}
    iv_rows = []
    for col in feature_cols:
        try:
            fit = wb.fit_feature_woe(train_df[col], train_df[target_col])
        except (ValueError, ZeroDivisionError):
            continue
        fits[col] = fit
        iv_rows.append({"feature_name": col, "iv_value": fit["iv_total"], "bin_count": fit["bin_count"]})
    iv_summary = pd.DataFrame(iv_rows).sort_values("iv_value", ascending=False).reset_index(drop=True)
    return fits, iv_summary


def run_baseline(df: pd.DataFrame, conn: sqlite3.Connection, min_iv: float = 0.02, top_n: int = 25,
                  test_size: float = 0.2, random_state: int = 42) -> dict:
    feature_cols = [c for c in df.columns if c not in (TARGET_COL, ID_COL)]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL]
    )

    fits, iv_summary = fit_woe_for_all_features(train_df, feature_cols)
    wb.save_iv_summary(iv_summary, conn)

    selected = wb.select_features_by_iv(iv_summary, min_iv=min_iv)[:top_n]
    if not selected:
        raise RuntimeError("IV 기준을 만족하는 변수가 없습니다. min_iv를 낮춰보세요.")

    woe_full = pd.DataFrame({c: wb.transform_feature_woe_values(df[c], fits[c]) for c in selected})
    woe_full[ID_COL] = df[ID_COL].values
    wb.save_features_woe(woe_full, fits, selected, conn, id_col=ID_COL)

    X_train = pd.DataFrame({c: wb.transform_feature_woe_values(train_df[c], fits[c]) for c in selected})
    X_test = pd.DataFrame({c: wb.transform_feature_woe_values(test_df[c], fits[c]) for c in selected})
    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba_test))
    ks = compute_ks(y_test, proba_test)
    gini = gini_from_auc(auc)

    cur = conn.execute(
        "INSERT INTO model_runs (model_type, train_date, auc, ks, gini, params_json) VALUES (?, ?, ?, ?, ?, ?)",
        ("logistic_woe", date.today().isoformat(), auc, ks, gini, json.dumps({"features": selected, "min_iv": min_iv})),
    )
    run_id = cur.lastrowid
    conn.commit()

    proba_full = model.predict_proba(woe_full[selected])[:, 1]

    return {
        "run_id": run_id,
        "model": model,
        "selected_features": selected,
        "iv_summary": iv_summary,
        "fits": fits,
        "auc": auc,
        "ks": ks,
        "gini": gini,
        "train_df": train_df,
        "test_df": test_df,
        "y_test": y_test,
        "proba_test": proba_test,
        "proba_full": proba_full,
        "ids_full": df[ID_COL].to_numpy(),
        "target_full": df[TARGET_COL].to_numpy(),
    }
