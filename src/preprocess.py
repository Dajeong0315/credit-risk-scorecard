"""Load raw Home Credit application data, clean known anomalies, store to SQLite."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "application_train.csv"

# DAYS_EMPLOYED uses 365243 as a sentinel for "not currently employed" (a known
# Home Credit Default Risk data quirk) -- treat it as missing rather than a real value.
ANOMALY_SENTINELS = {
    "DAYS_EMPLOYED": 365243,
}

# Drop columns that are missing for more than this fraction of rows.
MAX_NULL_FRAC = 0.6


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python src/download_data.py` 로 데이터를 다운로드하세요."
        )
    return pd.read_csv(path)


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, sentinel in ANOMALY_SENTINELS.items():
        if col in df.columns:
            df.loc[df[col] == sentinel, col] = pd.NA
    return df


def drop_sparse_columns(df: pd.DataFrame, max_null_frac: float = MAX_NULL_FRAC) -> pd.DataFrame:
    null_frac = df.isna().mean()
    keep_cols = null_frac[null_frac < max_null_frac].index
    return df[keep_cols]


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly cleaning + drop very sparse columns. Missing-value imputation is
    deliberately left to WoE binning, which treats missing as its own bin."""
    df = clean_anomalies(df)
    df = drop_sparse_columns(df)
    return df


def save_applicants(df: pd.DataFrame, conn: sqlite3.Connection, id_col: str = "SK_ID_CURR",
                     target_col: str = "TARGET") -> None:
    feature_cols = [c for c in df.columns if c not in (id_col, target_col)]
    ids = df[id_col].tolist()
    targets = df[target_col].tolist() if target_col in df.columns else [None] * len(df)
    # .values.tolist() is much faster than df.to_dict(orient="records") on a wide frame.
    value_rows = df[feature_cols].values.tolist()

    def clean(v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    rows = []
    for sk_id, target, values in zip(ids, targets, value_rows):
        record = {col: clean(v) for col, v in zip(feature_cols, values)}
        rows.append((int(sk_id), None if pd.isna(target) else int(target), json.dumps(record, default=str)))

    conn.execute("DELETE FROM applicants")
    conn.executemany(
        "INSERT INTO applicants (sk_id_curr, target, raw_features_json) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
