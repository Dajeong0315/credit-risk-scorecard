"""Feature engineering from auxiliary Home Credit tables.

application_train.csv alone omits a client's credit-bureau history and their
prior Home Credit loan behavior -- both strong predictors in practice (this
is the standard approach used by top Home Credit Default Risk Kaggle
solutions). This module aggregates each of the following down to one row per
SK_ID_CURR and merges them onto the main application dataframe:

- bureau.csv + bureau_balance.csv    (credit-bureau-reported loans elsewhere)
- previous_application.csv            (this client's previous Home Credit loans)
- POS_CASH_balance.csv                 (monthly POS/cash loan installment status)
- credit_card_balance.csv               (monthly credit card balance/drawings)
- installments_payments.csv              (actual vs. scheduled installment payments)

The latter three are keyed by SK_ID_PREV but also carry SK_ID_CURR directly,
so no intermediate join through previous_application is needed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BUREAU_DPD_STATUSES = {"1", "2", "3", "4", "5"}


def _bureau_balance_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    bb = pd.read_csv(raw_dir / "bureau_balance.csv")
    bb["IS_DPD"] = bb["STATUS"].isin(BUREAU_DPD_STATUSES).astype(int)
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


def build_pos_cash_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    pos = pd.read_csv(raw_dir / "POS_CASH_balance.csv")
    pos["IS_COMPLETED"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(int)

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "count"),
        POS_NUNIQUE_PREV=("SK_ID_PREV", "nunique"),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_COMPLETED_COUNT=("IS_COMPLETED", "sum"),
    ).reset_index()
    agg["POS_COMPLETED_RATIO"] = agg["POS_COMPLETED_COUNT"] / agg["POS_COUNT"]
    return agg


def build_credit_card_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    cc = pd.read_csv(raw_dir / "credit_card_balance.csv")
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)

    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "count"),
        CC_NUNIQUE_PREV=("SK_ID_PREV", "nunique"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_CNT_DRAWINGS_CURRENT_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
        CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
    ).reset_index()
    return agg


def build_installments_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    inst = pd.read_csv(raw_dir / "installments_payments.csv")
    days_late = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["DPD"] = days_late.clip(lower=0)
    inst["DBD"] = (-days_late).clip(lower=0)
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

    agg = inst.groupby("SK_ID_CURR").agg(
        INSTAL_COUNT=("SK_ID_PREV", "count"),
        INSTAL_DPD_MEAN=("DPD", "mean"),
        INSTAL_DPD_MAX=("DPD", "max"),
        INSTAL_DBD_MEAN=("DBD", "mean"),
        INSTAL_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INSTAL_PAYMENT_DIFF_SUM=("PAYMENT_DIFF", "sum"),
        INSTAL_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
    ).reset_index()
    return agg


def build_extended_features(base_df: pd.DataFrame, raw_dir: Path = RAW_DIR, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """Left-join bureau/previous-application/POS/credit-card/installments aggregates
    onto base_df. Applicants with no history in a given table simply get NaN in
    those columns -- WoE binning treats that as its own informative "Missing" bin."""
    merged = base_df
    for builder in (
        build_bureau_features,
        build_previous_application_features,
        build_pos_cash_features,
        build_credit_card_features,
        build_installments_features,
    ):
        merged = merged.merge(builder(raw_dir), on=id_col, how="left")
    return merged
