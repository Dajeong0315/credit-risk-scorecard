"""SQLite schema init and connection helper for the credit scoring pipeline."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "credit_scoring.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS applicants (
    sk_id_curr INTEGER PRIMARY KEY,
    target INTEGER,
    raw_features_json TEXT
);

CREATE TABLE IF NOT EXISTS features_woe (
    sk_id_curr INTEGER,
    feature_name TEXT,
    bin_label TEXT,
    woe_value REAL,
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr)
);

CREATE TABLE IF NOT EXISTS iv_summary (
    feature_name TEXT PRIMARY KEY,
    iv_value REAL,
    bin_count INTEGER
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT,
    train_date TEXT,
    auc REAL,
    ks REAL,
    gini REAL,
    params_json TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    run_id INTEGER,
    sk_id_curr INTEGER,
    score REAL,
    grade TEXT,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
);

CREATE TABLE IF NOT EXISTS cutoff_simulation (
    run_id INTEGER,
    cutoff REAL,
    approval_rate REAL,
    expected_default_rate REAL,
    expected_loss REAL,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
);

CREATE TABLE IF NOT EXISTS psi_monitoring (
    run_id INTEGER,
    target_name TEXT,
    psi_value REAL,
    period TEXT,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"initialized schema at {DB_PATH}")
