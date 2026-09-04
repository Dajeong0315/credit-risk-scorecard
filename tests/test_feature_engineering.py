import pandas as pd
import pytest

from src import feature_engineering as fe


@pytest.fixture
def raw_dir(tmp_path):
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [100, 101, 200],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
        "CREDIT_CURRENCY": ["currency 1"] * 3,
        "DAYS_CREDIT": [-100, -500, -50],
        "CREDIT_DAY_OVERDUE": [0, 5, 0],
        "DAYS_CREDIT_ENDDATE": [200, -10, 300],
        "DAYS_ENDDATE_FACT": [None, -10, None],
        "AMT_CREDIT_MAX_OVERDUE": [0, 100, 0],
        "CNT_CREDIT_PROLONG": [0, 1, 0],
        "AMT_CREDIT_SUM": [10000, 5000, 20000],
        "AMT_CREDIT_SUM_DEBT": [2000, 0, 8000],
        "AMT_CREDIT_SUM_LIMIT": [0, 0, 0],
        "AMT_CREDIT_SUM_OVERDUE": [0, 0, 0],
        "CREDIT_TYPE": ["Consumer credit"] * 3,
        "DAYS_CREDIT_UPDATE": [-10, -400, -5],
        "AMT_ANNUITY": [1000, 0, 2000],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [100, 100, 101],
        "MONTHS_BALANCE": [0, -1, 0],
        "STATUS": ["0", "1", "C"],
    })
    previous = pd.DataFrame({
        "SK_ID_PREV": [1000, 1001, 2000],
        "SK_ID_CURR": [1, 1, 2],
        "NAME_CONTRACT_TYPE": ["Cash loans"] * 3,
        "AMT_ANNUITY": [500, 600, 700],
        "AMT_APPLICATION": [10000, 20000, 15000],
        "AMT_CREDIT": [9000, 20000, 15000],
        "AMT_DOWN_PAYMENT": [0, 0, 0],
        "AMT_GOODS_PRICE": [10000, 20000, 15000],
        "WEEKDAY_APPR_PROCESS_START": ["MONDAY"] * 3,
        "HOUR_APPR_PROCESS_START": [10, 11, 12],
        "FLAG_LAST_APPL_PER_CONTRACT": ["Y"] * 3,
        "NFLAG_LAST_APPL_IN_DAY": [1, 1, 1],
        "RATE_DOWN_PAYMENT": [0, 0, 0],
        "RATE_INTEREST_PRIMARY": [None, None, None],
        "RATE_INTEREST_PRIVILEGED": [None, None, None],
        "NAME_CASH_LOAN_PURPOSE": ["XAP"] * 3,
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "DAYS_DECISION": [-30, -200, -60],
        "NAME_PAYMENT_TYPE": ["Cash"] * 3,
        "CODE_REJECT_REASON": ["XAP"] * 3,
        "NAME_TYPE_SUITE": [None] * 3,
        "NAME_CLIENT_TYPE": ["New"] * 3,
        "NAME_GOODS_CATEGORY": ["XNA"] * 3,
        "NAME_PORTFOLIO": ["POS"] * 3,
        "NAME_PRODUCT_TYPE": ["XNA"] * 3,
        "CHANNEL_TYPE": ["Country-wide"] * 3,
        "SELLERPLACE_AREA": [0, 0, 0],
        "NAME_SELLER_INDUSTRY": ["XNA"] * 3,
        "CNT_PAYMENT": [12, 24, 12],
        "NAME_YIELD_GROUP": ["middle"] * 3,
        "PRODUCT_COMBINATION": ["POS household"] * 3,
        "DAYS_FIRST_DRAWING": [None] * 3,
        "DAYS_FIRST_DUE": [None] * 3,
        "DAYS_LAST_DUE_1ST_VERSION": [None] * 3,
        "DAYS_LAST_DUE": [None] * 3,
        "DAYS_TERMINATION": [None] * 3,
        "NFLAG_INSURED_ON_APPROVAL": [0, 0, 0],
    })

    pos_cash = pd.DataFrame({
        "SK_ID_PREV": [1000, 1000, 2000],
        "SK_ID_CURR": [1, 1, 2],
        "MONTHS_BALANCE": [-2, -1, -1],
        "CNT_INSTALMENT": [12, 12, 12],
        "CNT_INSTALMENT_FUTURE": [10, 9, 11],
        "NAME_CONTRACT_STATUS": ["Active", "Completed", "Active"],
        "SK_DPD": [0, 0, 5],
        "SK_DPD_DEF": [0, 0, 0],
    })
    credit_card = pd.DataFrame({
        "SK_ID_PREV": [1001, 1001],
        "SK_ID_CURR": [1, 1],
        "MONTHS_BALANCE": [-1, -2],
        "AMT_BALANCE": [5000, 4000],
        "AMT_CREDIT_LIMIT_ACTUAL": [10000, 10000],
        "AMT_DRAWINGS_ATM_CURRENT": [0, 0],
        "AMT_DRAWINGS_CURRENT": [0, 0],
        "AMT_DRAWINGS_OTHER_CURRENT": [0, 0],
        "AMT_DRAWINGS_POS_CURRENT": [0, 0],
        "AMT_INST_MIN_REGULARITY": [200, 200],
        "AMT_PAYMENT_CURRENT": [500, 500],
        "AMT_PAYMENT_TOTAL_CURRENT": [500, 500],
        "AMT_RECEIVABLE_PRINCIPAL": [5000, 4000],
        "AMT_RECIVABLE": [5000, 4000],
        "AMT_TOTAL_RECEIVABLE": [5000, 4000],
        "CNT_DRAWINGS_ATM_CURRENT": [0, 0],
        "CNT_DRAWINGS_CURRENT": [1, 1],
        "CNT_DRAWINGS_OTHER_CURRENT": [0, 0],
        "CNT_DRAWINGS_POS_CURRENT": [1, 1],
        "CNT_INSTALMENT_MATURE_CUM": [5, 4],
        "NAME_CONTRACT_STATUS": ["Active", "Active"],
        "SK_DPD": [0, 0],
        "SK_DPD_DEF": [0, 0],
    })
    installments = pd.DataFrame({
        "SK_ID_PREV": [1000, 1000, 2000],
        "SK_ID_CURR": [1, 1, 2],
        "NUM_INSTALMENT_VERSION": [1, 1, 1],
        "NUM_INSTALMENT_NUMBER": [1, 2, 1],
        "DAYS_INSTALMENT": [-100, -70, -50],
        "DAYS_ENTRY_PAYMENT": [-102, -65, -50],
        "AMT_INSTALMENT": [1000, 1000, 2000],
        "AMT_PAYMENT": [1000, 1000, 1800],
    })

    bureau.to_csv(tmp_path / "bureau.csv", index=False)
    bureau_balance.to_csv(tmp_path / "bureau_balance.csv", index=False)
    previous.to_csv(tmp_path / "previous_application.csv", index=False)
    pos_cash.to_csv(tmp_path / "POS_CASH_balance.csv", index=False)
    credit_card.to_csv(tmp_path / "credit_card_balance.csv", index=False)
    installments.to_csv(tmp_path / "installments_payments.csv", index=False)
    return tmp_path


def test_build_bureau_features_aggregates_per_applicant(raw_dir):
    result = fe.build_bureau_features(raw_dir).set_index("SK_ID_CURR")
    assert result.loc[1, "BUREAU_COUNT"] == 2
    assert result.loc[1, "BUREAU_ACTIVE_COUNT"] == 1
    assert result.loc[1, "BUREAU_ACTIVE_RATIO"] == pytest.approx(0.5)
    assert result.loc[2, "BUREAU_COUNT"] == 1
    assert result.loc[2, "BUREAU_ACTIVE_RATIO"] == pytest.approx(1.0)


def test_build_previous_application_features_aggregates_per_applicant(raw_dir):
    result = fe.build_previous_application_features(raw_dir).set_index("SK_ID_CURR")
    assert result.loc[1, "PREV_COUNT"] == 2
    assert result.loc[1, "PREV_APPROVED_COUNT"] == 1
    assert result.loc[1, "PREV_REFUSED_COUNT"] == 1
    assert result.loc[1, "PREV_APPROVED_RATIO"] == pytest.approx(0.5)
    assert result.loc[2, "PREV_APPROVED_RATIO"] == pytest.approx(1.0)


def test_build_pos_cash_features_aggregates_per_applicant(raw_dir):
    result = fe.build_pos_cash_features(raw_dir).set_index("SK_ID_CURR")
    assert result.loc[1, "POS_COUNT"] == 2
    assert result.loc[1, "POS_COMPLETED_COUNT"] == 1
    assert result.loc[1, "POS_COMPLETED_RATIO"] == pytest.approx(0.5)
    assert result.loc[2, "POS_SK_DPD_MAX"] == 5


def test_build_credit_card_features_computes_utilization(raw_dir):
    result = fe.build_credit_card_features(raw_dir).set_index("SK_ID_CURR")
    assert result.loc[1, "CC_COUNT"] == 2
    # balances 5000/4000 over a 10000 limit -> utilization 0.5 / 0.4, mean 0.45
    assert result.loc[1, "CC_UTILIZATION_MEAN"] == pytest.approx(0.45)
    assert result.loc[1, "CC_UTILIZATION_MAX"] == pytest.approx(0.5)


def test_build_installments_features_computes_dpd_and_payment_ratio(raw_dir):
    result = fe.build_installments_features(raw_dir).set_index("SK_ID_CURR")
    assert result.loc[1, "INSTAL_COUNT"] == 2
    # first installment paid 2 days early (-102 vs due -100) -> DPD 0, DBD 2
    # second installment paid 5 days late (-65 vs due -70) -> DPD 5, DBD 0
    # DPD mean = (0 + 5) / 2 = 2.5
    assert result.loc[1, "INSTAL_DPD_MEAN"] == pytest.approx(2.5)
    assert result.loc[1, "INSTAL_PAYMENT_RATIO_MEAN"] == pytest.approx(1.0)
    # applicant 2 underpaid: 1800 paid vs 2000 due
    assert result.loc[2, "INSTAL_PAYMENT_RATIO_MEAN"] == pytest.approx(0.9)
    assert result.loc[2, "INSTAL_PAYMENT_DIFF_SUM"] == pytest.approx(200)


def test_build_extended_features_left_join_keeps_all_applicants(raw_dir):
    base_df = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]})
    merged = fe.build_extended_features(base_df, raw_dir)
    assert len(merged) == 3
    # applicant 3 has no history anywhere -> NaN, not dropped
    assert merged.loc[merged["SK_ID_CURR"] == 3, "BUREAU_COUNT"].isna().all()
    assert merged.loc[merged["SK_ID_CURR"] == 3, "POS_COUNT"].isna().all()
    assert merged.loc[merged["SK_ID_CURR"] == 3, "CC_COUNT"].isna().all()
    assert merged.loc[merged["SK_ID_CURR"] == 3, "INSTAL_COUNT"].isna().all()
    assert merged.loc[merged["SK_ID_CURR"] == 1, "BUREAU_COUNT"].iloc[0] == 2
