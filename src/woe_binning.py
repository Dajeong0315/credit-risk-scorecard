"""Weight of Evidence (WoE) binning and Information Value (IV) calculation.

Standard credit-scorecard definitions:
    WoE_bin = ln( (bad_bin / total_bad) / (good_bin / total_good) )
    IV_bin  = (bad_bin / total_bad - good_bin / total_good) * WoE_bin
    IV_total = sum(IV_bin)

"bad" == target == 1 (default), "good" == target == 0.
"""
import sqlite3

import numpy as np
import pandas as pd

EPS = 1e-6


def woe_iv_from_counts(bad_count: float, good_count: float, total_bad: float, total_good: float) -> tuple:
    """Pure function: WoE and IV contribution for a single bin given raw counts."""
    bad_dist = max(bad_count, EPS) / total_bad
    good_dist = max(good_count, EPS) / total_good
    woe = float(np.log(bad_dist / good_dist))
    iv = float((bad_dist - good_dist) * woe)
    return woe, iv


def calculate_woe_iv(bin_labels: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Given per-row bin labels and target (0/1), return one row per bin with woe/iv."""
    df = pd.DataFrame({"bin_label": bin_labels.values, "target": target.values})
    total_bad = float(df["target"].sum())
    total_good = float((df["target"] == 0).sum())
    if total_bad == 0 or total_good == 0:
        raise ValueError("target must contain both classes (0 and 1) to compute WoE/IV")

    grouped = df.groupby("bin_label")["target"].agg(total="count", bad_count="sum").reset_index()
    grouped["good_count"] = grouped["total"] - grouped["bad_count"]

    woes, ivs = [], []
    for _, row in grouped.iterrows():
        woe, iv = woe_iv_from_counts(row["bad_count"], row["good_count"], total_bad, total_good)
        woes.append(woe)
        ivs.append(iv)
    grouped["woe"] = woes
    grouped["iv"] = ivs
    return grouped


def _apply_numeric_edges(series: pd.Series, edges: list) -> pd.Series:
    non_null = series.dropna()
    cut = pd.cut(non_null, bins=edges, include_lowest=True, duplicates="drop")
    labels = cut.astype(str)
    result = pd.Series("Missing", index=series.index, dtype=object)
    result.loc[non_null.index] = labels.values
    return result


def bin_numeric(series: pd.Series, max_bins: int = 10) -> tuple:
    """Quantile-bin a numeric series. Returns (bin_label_series, edges|None)."""
    non_null = series.dropna()
    if non_null.nunique() <= 1:
        labels = pd.Series(np.where(series.isna(), "Missing", "All"), index=series.index)
        return labels, None

    edges = None
    n_bins = max_bins
    while n_bins >= 2:
        try:
            _, candidate_edges = pd.qcut(non_null, q=n_bins, duplicates="drop", retbins=True)
            if len(candidate_edges) >= 3:
                edges = list(candidate_edges)
                break
        except ValueError:
            pass
        n_bins -= 1

    if edges is None:
        labels = pd.Series(np.where(series.isna(), "Missing", "All"), index=series.index)
        return labels, None

    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = _apply_numeric_edges(series, edges)
    return labels, edges


def bin_categorical(series: pd.Series, min_bin_pct: float = 0.01) -> tuple:
    """Bin a categorical series, grouping rare categories into 'Other'. Returns (bin_label_series, kept_categories)."""
    filled = series.fillna("Missing").astype(str)
    freq = filled.value_counts(normalize=True)
    keep_categories = set(freq[freq >= min_bin_pct].index) - {"Missing"}
    labels = filled.where(filled.isin(keep_categories) | (filled == "Missing"), "Other")
    return labels, keep_categories


LOW_CARDINALITY_THRESHOLD = 10


def fit_feature_woe(series: pd.Series, target: pd.Series, max_bins: int = 10, min_bin_pct: float = 0.01,
                     feature_type: str = "auto") -> dict:
    """Fit WoE bins for one feature against target on a training set. Returns a fit dict
    that can be reused with transform_feature_woe on any other dataframe (e.g. test set).

    Numeric features with few distinct values (e.g. binary FLAG_* columns) are binned
    discretely (one bin per value) rather than by quantile, so they don't collapse into
    a single uninformative bin.
    """
    if feature_type == "auto":
        feature_type = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"

    is_low_card_numeric = feature_type == "numeric" and series.dropna().nunique() <= LOW_CARDINALITY_THRESHOLD

    if feature_type == "numeric" and not is_low_card_numeric:
        bin_labels, edges = bin_numeric(series, max_bins=max_bins)
        categories = None
        mode = "quantile"
    else:
        bin_labels, categories = bin_categorical(series, min_bin_pct=0.0 if is_low_card_numeric else min_bin_pct)
        edges = None
        mode = "discrete"

    woe_table = calculate_woe_iv(bin_labels, target)
    iv_total = float(woe_table["iv"].sum())
    woe_map = dict(zip(woe_table["bin_label"], woe_table["woe"]))

    return {
        "feature_name": series.name,
        "feature_type": feature_type,
        "mode": mode,
        "edges": edges,
        "categories": categories,
        "woe_map": woe_map,
        "woe_table": woe_table,
        "iv_total": iv_total,
        "bin_count": int(len(woe_table)),
    }


def transform_feature_woe(series: pd.Series, fit_result: dict) -> pd.Series:
    """Apply a fitted WoE mapping to (possibly new/unseen) data."""
    if fit_result["mode"] == "quantile":
        if fit_result["edges"] is None:
            bin_labels = pd.Series(np.where(series.isna(), "Missing", "All"), index=series.index)
        else:
            bin_labels = _apply_numeric_edges(series, fit_result["edges"])
    else:
        filled = series.fillna("Missing").astype(str)
        categories = fit_result["categories"] or set()
        bin_labels = filled.where(filled.isin(categories) | (filled == "Missing"), "Other")

    return bin_labels.map(fit_result["woe_map"]).fillna(0.0), bin_labels


def transform_feature_woe_values(series: pd.Series, fit_result: dict) -> pd.Series:
    """Like transform_feature_woe but returns only the numeric WoE values."""
    woe_values, _ = transform_feature_woe(series, fit_result)
    return woe_values


def select_features_by_iv(iv_summary: pd.DataFrame, min_iv: float = 0.02, max_iv: float = 0.5) -> list:
    """Standard scorecard IV screening rule: <0.02 not useful, >0.5 suspiciously high (possible leakage)."""
    mask = (iv_summary["iv_value"] >= min_iv) & (iv_summary["iv_value"] <= max_iv)
    return iv_summary.loc[mask].sort_values("iv_value", ascending=False)["feature_name"].tolist()


def save_iv_summary(iv_summary: pd.DataFrame, conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM iv_summary")
    rows = list(iv_summary[["feature_name", "iv_value", "bin_count"]].itertuples(index=False, name=None))
    conn.executemany("INSERT OR REPLACE INTO iv_summary (feature_name, iv_value, bin_count) VALUES (?, ?, ?)", rows)
    conn.commit()


def save_features_woe(df: pd.DataFrame, fits: dict, selected_features: list, conn: sqlite3.Connection,
                       id_col: str = "SK_ID_CURR") -> None:
    """Store per-applicant bin_label/woe_value for each selected feature."""
    conn.execute("DELETE FROM features_woe")
    for feature in selected_features:
        fit = fits[feature]
        woe_values, bin_labels = transform_feature_woe(df[feature], fit)
        rows = list(zip(df[id_col], [feature] * len(df), bin_labels, woe_values))
        conn.executemany(
            "INSERT INTO features_woe (sk_id_curr, feature_name, bin_label, woe_value) VALUES (?, ?, ?, ?)",
            rows,
        )
    conn.commit()
