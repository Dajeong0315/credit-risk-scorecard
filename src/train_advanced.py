"""Advanced model: LightGBM on raw (non-WoE) features, for champion-challenger comparison."""
import json
from datetime import date

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.train_baseline import compute_ks, gini_from_auc, TARGET_COL, ID_COL

# AUC improvement over baseline required for LightGBM to be adopted as champion.
CHAMPION_AUC_MARGIN = 0.01


def run_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, conn, random_state: int = 42) -> dict:
    feature_cols = [c for c in train_df.columns if c not in (TARGET_COL, ID_COL)]
    # pandas >= 2.3 may store text columns as a "str" dtype (future.infer_string)
    # instead of "object", so detect categoricals as "not numeric" rather than
    # checking for a specific dtype label.
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    proba_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba_test))
    ks = compute_ks(y_test, proba_test)
    gini = gini_from_auc(auc)

    cur = conn.execute(
        "INSERT INTO model_runs (model_type, train_date, auc, ks, gini, params_json) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "lightgbm",
            date.today().isoformat(),
            auc,
            ks,
            gini,
            json.dumps({"n_estimators": model.best_iteration_ or model.n_estimators, "features": feature_cols}),
        ),
    )
    run_id = cur.lastrowid
    conn.commit()

    return {
        "run_id": run_id,
        "model": model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "auc": auc,
        "ks": ks,
        "gini": gini,
        "proba_test": proba_test,
        "y_test": y_test,
    }


def score_full_population(model, df: pd.DataFrame, feature_cols: list, cat_cols: list) -> pd.Series:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index)


def choose_champion(baseline_result: dict, advanced_result: dict, margin: float = CHAMPION_AUC_MARGIN) -> dict:
    """Champion-challenger decision: advanced model wins only if it beats the
    baseline logistic scorecard by more than `margin` AUC -- otherwise keep the
    baseline for its interpretability advantage."""
    auc_gain = advanced_result["auc"] - baseline_result["auc"]
    if auc_gain > margin:
        champion = "lightgbm"
        reason = (
            f"LightGBM이 베이스라인 대비 AUC +{auc_gain:.4f}pt 우세 (기준 {margin} 초과) -> 챔피언으로 채택"
        )
    else:
        champion = "logistic_woe"
        reason = (
            f"LightGBM의 AUC 개선폭이 +{auc_gain:.4f}pt로 기준({margin}) 이하 -> "
            "해석 가능성이 높은 로지스틱 WoE 스코어카드를 챔피언으로 유지"
        )
    return {"champion": champion, "auc_gain": auc_gain, "reason": reason}
