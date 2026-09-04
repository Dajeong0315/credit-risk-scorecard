"""Export a small, deployable subset of the SQLite results for the Streamlit dashboard.

app.py only ever queries model_runs, iv_summary, cutoff_simulation, psi_monitoring,
and scores -- never the much larger features_woe (per-applicant WoE rows) or
applicants (raw per-applicant JSON) tables. The full db/credit_scoring.db can be
1.5GB+, far too large to commit to GitHub or run on Streamlit Community Cloud's
free tier, so this script copies just the tables the dashboard needs into a
separate db/dashboard.db that IS committed to the repo.

Usage:
    python src/export_dashboard_db.py
"""
import sqlite3
from pathlib import Path

SOURCE_DB = Path(__file__).resolve().parent.parent / "db" / "credit_scoring.db"
DASHBOARD_DB = Path(__file__).resolve().parent.parent / "db" / "dashboard.db"

TABLES = ["model_runs", "iv_summary", "cutoff_simulation", "psi_monitoring", "scores"]


def export_dashboard_db(source: Path = SOURCE_DB, dest: Path = DASHBOARD_DB) -> None:
    if not source.exists():
        raise FileNotFoundError(f"{source} 가 없습니다. 먼저 `python run.py` 를 실행하세요.")

    dest.unlink(missing_ok=True)
    src_conn = sqlite3.connect(source)
    dest_conn = sqlite3.connect(dest)

    schema_rows = src_conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ({})".format(
            ",".join("?" for _ in TABLES)
        ),
        TABLES,
    ).fetchall()
    schema_by_name = dict(schema_rows)

    for table in TABLES:
        dest_conn.execute(schema_by_name[table])
        rows = src_conn.execute(f"SELECT * FROM {table}").fetchall()
        if rows:
            placeholders = ",".join("?" for _ in rows[0])
            dest_conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)

    dest_conn.commit()
    dest_conn.execute("VACUUM")
    dest_conn.close()
    src_conn.close()

    size_kb = dest.stat().st_size / 1024
    print(f"[OK] {dest} 생성됨 ({size_kb:.1f} KB)")


if __name__ == "__main__":
    export_dashboard_db()
