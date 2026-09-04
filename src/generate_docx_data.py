"""Extract key summary figures from the SQLite results into JSON, for the
interview/cover-letter summary document generator (scripts/build_summary_docx.js)."""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "credit_scoring.db"
OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "summary_data.json"


def extract_summary(db_path: Path = DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    model_runs = [dict(r) for r in conn.execute("SELECT * FROM model_runs ORDER BY run_id").fetchall()]
    baseline = next(r for r in reversed(model_runs) if r["model_type"] == "logistic_woe")
    advanced = next(r for r in reversed(model_runs) if r["model_type"] == "lightgbm")

    auc_gain = advanced["auc"] - baseline["auc"]
    champion = "lightgbm" if auc_gain > 0.01 else "logistic_woe"
    champion_run = advanced if champion == "lightgbm" else baseline

    grade_rows = conn.execute(
        "SELECT s.grade AS grade, COUNT(*) AS n, AVG(s.score) AS avg_score, "
        "AVG(CASE WHEN a.target = 1 THEN 1.0 ELSE 0.0 END) AS default_rate "
        "FROM scores s JOIN applicants a ON a.sk_id_curr = s.sk_id_curr "
        "WHERE s.run_id = ? GROUP BY s.grade",
        (champion_run["run_id"],),
    ).fetchall()
    grade_stats = {
        r["grade"]: {"count": r["n"], "avg_score": r["avg_score"], "default_rate": r["default_rate"]}
        for r in grade_rows
    }

    cutoff_rows = conn.execute(
        "SELECT * FROM cutoff_simulation WHERE run_id = ? ORDER BY approval_rate", (champion_run["run_id"],)
    ).fetchall()
    cutoff_50 = min(cutoff_rows, key=lambda r: abs(r["approval_rate"] - 0.5)) if cutoff_rows else None

    iv_top = conn.execute("SELECT * FROM iv_summary ORDER BY iv_value DESC LIMIT 5").fetchall()

    psi_score = conn.execute(
        "SELECT * FROM psi_monitoring WHERE run_id = ? AND target_name = 'score'", (champion_run["run_id"],)
    ).fetchone()

    n_applicants = conn.execute("SELECT COUNT(*) AS n FROM applicants").fetchone()["n"]

    conn.close()

    grade_a = grade_stats.get("A", {}).get("default_rate")
    grade_e = grade_stats.get("E", {}).get("default_rate")
    grade_ratio = (grade_e / grade_a) if grade_a else None

    summary = {
        "n_applicants": n_applicants,
        "baseline": {"model_type": baseline["model_type"], "auc": baseline["auc"], "ks": baseline["ks"], "gini": baseline["gini"]},
        "advanced": {"model_type": advanced["model_type"], "auc": advanced["auc"], "ks": advanced["ks"], "gini": advanced["gini"]},
        "champion": champion,
        "auc_gain": auc_gain,
        "grade_stats": grade_stats,
        "grade_default_ratio_e_over_a": grade_ratio,
        "cutoff_50": dict(cutoff_50) if cutoff_50 else None,
        "iv_top": [dict(r) for r in iv_top],
        "psi_score": dict(psi_score) if psi_score else None,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    data = extract_summary()
    print(json.dumps(data, ensure_ascii=False, indent=2))
