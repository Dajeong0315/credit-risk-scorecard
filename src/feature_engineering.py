"""Feature engineering from auxiliary Home Credit tables.

application_train.csv alone omits a client's credit-bureau history and their
prior Home Credit application history -- both strong predictors in practice
(this is the standard approach used by top Home Credit Default Risk Kaggle
solutions). This module aggregates:

- bureau.csv + bureau_balance.csv  (credit-bureau-reported loans elsewhere)
- previous_application.csv          (this client's previous Home Credit loans)

down to one row per SK_ID_CURR and merges them onto the main application
dataframe. POS_CASH_balance.csv / credit_card_balance.csv /
installments_payments.csv (SK_ID_PREV-keyed, needing a second rollup level)
are deliberately out of scope for this pass -- see PROGRESS.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DPD_STATUSES = {"1", "2", "3", "4", "5"}


def _bureau_balance_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    bb = pd.read_csv(raw_dir / "bureau_balance.csv")
    bb["IS_DPD"] = bb["STATUS"].isin(DPD_STATUSES).astype(int)
    agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BB_DPD_COUNT=("IS_DPD", "sum"),
    ).reset_index()
    agg["BB_DPD_RATIO"] = agg["BB_DPD_COUNT"] / agg["BB_MONTHS_COUNT"].replace(0, np.nan)
    return agg


def build_bureau_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    bureau = pd.read_csv(raw_dir / "bureau.csv")
    bureau = bureau.merge(_bureau_balance_features(raw_dir), on="SK_ID_BUREAU", how="left")

    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["DEBT_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_AMT_CREDIT_SUM_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_CNT_CREDIT_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_DEBT_CREDIT_RATIO_MEAN=("DEBT_CREDIT_RATIO", "mean"),
        BUREAU_BB_DPD_COUNT_SUM=("BB_DPD_COUNT", "sum"),
        BUREAU_BB_DPD_RATIO_MEAN=("BB_DPD_RATIO", "mean"),
    ).reset_index()
    agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_COUNT"]
    return agg


def build_previous_application_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    prev = pd.read_csv(raw_dir / "previous_application.csv")
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"].replace(0, np.nan)

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
    ).reset_index()
    agg["PREV_APPROVED_RATIO"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_COUNT"]
    agg["PREV_REFUSED_RATIO"] = agg["PREV_REFUSED_COUNT"] / agg["PREV_COUNT"]
    return agg


def build_extended_features(base_df: pd.DataFrame, raw_dir: Path = RAW_DIR, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """Left-join bureau + previous_application aggregates onto base_df.
    Applicants with no bureau/previous-application history simply get NaN in
    the new columns -- WoE binning treats that as its own informative "Missing" bin."""
    merged = base_df.merge(build_bureau_features(raw_dir), on=id_col, how="left")
    merged = merged.merge(build_previous_application_features(raw_dir), on=id_col, how="left")
    return merged
