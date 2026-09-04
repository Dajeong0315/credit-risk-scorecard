"""Streamlit dashboard: interactive cutoff simulation over the SQLite results.

Run: streamlit run app.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_DASHBOARD_DB = Path(__file__).resolve().parent / "db" / "dashboard.db"
_FULL_DB = Path(__file__).resolve().parent / "db" / "credit_scoring.db"
# Prefer the small, deployable dashboard.db (exported via src/export_dashboard_db.py)
# when present; fall back to the full local db produced directly by run.py.
DB_PATH = _DASHBOARD_DB if _DASHBOARD_DB.exists() else _FULL_DB

# Colors from the project's validated categorical/status palette (dataviz skill reference).
COLOR_BLUE = "#2a78d6"
COLOR_RED = "#e34948"
COLOR_ORANGE = "#eb6834"
COLOR_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRADE_COLORS = {"A": "#1baf7a", "B": "#2a78d6", "C": "#eda100", "D": "#eb6834", "E": "#e34948"}
GRADE_ORDER = ["A", "B", "C", "D", "E"]

st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")


TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#3a3a3a"
GRIDLINE = "#e1e0d9"


def _chart_layout(fig, height=420, legend=None, margin=None, **kwargs):
    legend_style = {"font": dict(color=TEXT_PRIMARY, size=13)}
    if legend:
        legend_style.update(legend)
    fig.update_layout(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY, size=14),
        legend=legend_style,
        height=height,
        margin=margin or dict(l=10, r=10, t=30, b=10),
        **kwargs,
    )
    fig.update_xaxes(color=TEXT_SECONDARY, gridcolor=GRIDLINE, tickfont=dict(size=13))
    fig.update_yaxes(color=TEXT_SECONDARY, gridcolor=GRIDLINE, tickfont=dict(size=13))
    return fig


@st.cache_data
def load_static_tables():
    if not DB_PATH.exists():
        return None, None
    conn = sqlite3.connect(DB_PATH)
    model_runs = pd.read_sql("SELECT * FROM model_runs ORDER BY run_id", conn)
    iv_summary = pd.read_sql("SELECT * FROM iv_summary ORDER BY iv_value DESC", conn)
    conn.close()
    return model_runs, iv_summary


@st.cache_data
def load_run_tables(run_id: int):
    conn = sqlite3.connect(DB_PATH)
    scores = pd.read_sql("SELECT * FROM scores WHERE run_id = ?", conn, params=(run_id,))
    cutoff_df = pd.read_sql(
        "SELECT * FROM cutoff_simulation WHERE run_id = ? ORDER BY cutoff", conn, params=(run_id,)
    )
    psi_df = pd.read_sql("SELECT * FROM psi_monitoring WHERE run_id = ?", conn, params=(run_id,))
    conn.close()
    return scores, cutoff_df, psi_df


st.title("Credit Risk Scorecard 대시보드")
st.caption("Home Credit Default Risk · WoE 스코어카드 vs LightGBM 챔피언-챌린저")

if not DB_PATH.exists():
    st.error(f"{DB_PATH} 가 없습니다. 먼저 `python run.py` 를 실행하세요.")
    st.stop()

model_runs, iv_summary = load_static_tables()
if model_runs is None or model_runs.empty:
    st.error("model_runs 테이블이 비어 있습니다. `python run.py` 를 먼저 실행하세요.")
    st.stop()

runs_indexed = model_runs.set_index("run_id")
run_id = st.sidebar.selectbox(
    "모델 실행(run) 선택",
    options=list(runs_indexed.index[::-1]),
    format_func=lambda rid: f"#{rid} {runs_indexed.loc[rid, 'model_type']} (AUC={runs_indexed.loc[rid, 'auc']:.4f})",
)
run_row = runs_indexed.loc[run_id]
scores, cutoff_df, psi_df = load_run_tables(run_id)

m1, m2, m3 = st.columns(3)
m1.metric("AUC", f"{run_row['auc']:.4f}")
m2.metric("KS", f"{run_row['ks']:.4f}")
m3.metric("Gini", f"{run_row['gini']:.4f}")

st.divider()
st.subheader("컷오프 시뮬레이션")

if cutoff_df.empty:
    st.info("이 run에는 컷오프 시뮬레이션 데이터가 없습니다.")
else:
    min_c, max_c = float(cutoff_df["cutoff"].min()), float(cutoff_df["cutoff"].max())
    cutoff_val = st.slider(
        "컷오프 점수 (이 점수 이상만 승인)",
        min_value=round(min_c, 1),
        max_value=round(max_c, 1),
        value=round((min_c + max_c) / 2, 1),
        step=0.5,
    )
    nearest_idx = (cutoff_df["cutoff"] - cutoff_val).abs().idxmin()
    nearest = cutoff_df.loc[nearest_idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("승인율", f"{nearest['approval_rate']:.1%}")
    c2.metric("예상 부도율", f"{nearest['expected_default_rate']:.2%}")
    c3.metric("예상 손실률", f"{nearest['expected_loss']:.2%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cutoff_df["cutoff"], y=cutoff_df["approval_rate"],
                              name="승인율", mode="lines", line=dict(color=COLOR_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=cutoff_df["cutoff"], y=cutoff_df["expected_default_rate"],
                              name="예상 부도율", mode="lines", line=dict(color=COLOR_RED, width=2)))
    fig.add_trace(go.Scatter(x=cutoff_df["cutoff"], y=cutoff_df["expected_loss"],
                              name="예상 손실률", mode="lines", line=dict(color=COLOR_ORANGE, width=2)))
    fig.add_vline(x=cutoff_val, line_dash="dash", line_color=COLOR_MUTED)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title="컷오프 점수")
    _chart_layout(fig, height=460, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("등급별 분포 (A-E)")
if scores.empty:
    st.info("이 run에는 등급 데이터가 없습니다.")
else:
    grade_stats = (
        scores.groupby("grade")
        .agg(count=("sk_id_curr", "size"))
        .reindex(GRADE_ORDER)
        .reset_index()
    )
    fig2 = px.bar(grade_stats, x="grade", y="count", color="grade", text="count",
                   color_discrete_map=GRADE_COLORS, category_orders={"grade": GRADE_ORDER})
    fig2.update_traces(textposition="outside", textfont=dict(color=TEXT_PRIMARY, size=13))
    fig2.update_layout(showlegend=False)
    _chart_layout(fig2, height=380)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("IV 상위 변수 (Top 15)")
top_iv = iv_summary.head(15).sort_values("iv_value")
fig3 = px.bar(top_iv, x="iv_value", y="feature_name", orientation="h",
               color_discrete_sequence=[COLOR_BLUE], text="iv_value")
fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                    textfont=dict(color=TEXT_PRIMARY, size=12))
fig3.update_yaxes(title=None)
fig3.update_xaxes(range=[0, top_iv["iv_value"].max() * 1.3])
_chart_layout(fig3, height=520, margin=dict(l=10, r=60, t=30, b=10))
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("PSI 안정성 점검")
if psi_df.empty:
    st.info("이 run에는 PSI 데이터가 없습니다.")
else:
    psi_display = psi_df[["target_name", "period", "psi_value"]].copy()
    psi_display["해석"] = psi_display["psi_value"].apply(
        lambda v: "안정" if v < 0.1 else ("중간 변화" if v < 0.25 else "유의미한 변화")
    )
    st.dataframe(psi_display, use_container_width=True, hide_index=True)

shap_path = Path(__file__).resolve().parent / "outputs" / "shap_summary.png"
if shap_path.exists():
    st.divider()
    st.subheader("SHAP 요약 (챔피언 모델)")
    st.image(str(shap_path), use_container_width=True)
