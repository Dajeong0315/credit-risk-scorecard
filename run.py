"""MVP pipeline entry point.

Runs: preprocess -> WoE/IV -> baseline logistic scorecard -> LightGBM ->
champion-challenger -> score scaling & grading -> cutoff simulation -> REPORT.md.

Usage:
    python run.py
"""
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src import preprocess, psi, scoring, shap_explain, simulation, train_advanced, train_baseline
from src import woe_binning as wb
from src.db import get_connection, init_db

pd.set_option("display.width", 120)


def build_report(baseline: dict, advanced: dict, champion: dict, sim_df: pd.DataFrame,
                  grade_stats: pd.DataFrame, n_rows: int, n_features_selected: int,
                  psi_rows: list, shap_image_path: Path = None) -> str:
    iv_top = baseline["iv_summary"].head(15).copy()
    iv_top["iv_value"] = iv_top["iv_value"].round(4)

    lines = []
    lines.append("# credit-risk-scorecard 분석 리포트\n")
    lines.append(f"생성일: {date.today().isoformat()}\n")

    lines.append("## 1. 데이터 개요\n")
    lines.append(f"- 데이터: Home Credit Default Risk `application_train.csv`, {n_rows:,}건")
    lines.append(f"- 전처리 후 사용 변수 수(ID/TARGET 제외): {baseline['iv_summary'].shape[0]}")
    lines.append(f"- IV 기준(0.02~0.5) 선택된 변수 수: {n_features_selected}\n")

    lines.append("## 2. IV(Information Value) 상위 변수\n")
    lines.append("| feature_name | iv_value | bin_count |")
    lines.append("|---|---|---|")
    for _, r in iv_top.iterrows():
        lines.append(f"| {r['feature_name']} | {r['iv_value']} | {int(r['bin_count'])} |")
    lines.append("")

    lines.append("## 3. 모델 성능 비교 (챔피언-챌린저)\n")
    lines.append("| model | AUC | KS | Gini |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 로지스틱 (WoE 스코어카드, 베이스라인) | {baseline['auc']:.4f} | {baseline['ks']:.4f} | {baseline['gini']:.4f} |")
    lines.append(f"| LightGBM (고도화 모델) | {advanced['auc']:.4f} | {advanced['ks']:.4f} | {advanced['gini']:.4f} |")
    lines.append("")
    lines.append(f"**챔피언 모델: `{champion['champion']}`**\n")
    lines.append(f"- AUC 개선폭: {champion['auc_gain']:+.4f}")
    lines.append(f"- 판단 사유: {champion['reason']}\n")

    lines.append("## 4. 점수 스케일링 및 등급 설계\n")
    lines.append(
        f"- PDO(Points to Double Odds) 방식: base_score={scoring.DEFAULT_BASE_SCORE}, "
        f"base_odds={scoring.DEFAULT_BASE_ODDS}:1, pdo={scoring.DEFAULT_PDO}"
    )
    lines.append("- 등급: A(최우량, 저위험) ~ E(최고위험), 동일 인원수(quantile) 기준 5등급\n")
    lines.append("| grade | count | avg_score | default_rate |")
    lines.append("|---|---|---|---|")
    for _, r in grade_stats.iterrows():
        lines.append(f"| {r['grade']} | {int(r['count'])} | {r['avg_score']:.1f} | {r['default_rate']:.4f} |")
    lines.append("")

    lines.append("## 5. 컷오프 시뮬레이션\n")
    lines.append(
        f"- 가정: LGD(Loss Given Default) = {simulation.DEFAULT_LGD:.0%} "
        "(Basel 무담보 소매여신 표준 가정)\n"
    )
    lines.append("| cutoff score | approval_rate | expected_default_rate | expected_loss |")
    lines.append("|---|---|---|---|")
    for _, r in sim_df.iterrows():
        lines.append(
            f"| {r['cutoff']:.1f} | {r['approval_rate']:.2%} | {r['expected_default_rate']:.2%} | {r['expected_loss']:.2%} |"
        )
    lines.append("")

    lines.append("## 6. PSI(Population Stability Index) 안정성 점검\n")
    lines.append(
        "- 기준(expected)=train, 비교 대상(actual)=test 분포. 해석: PSI<0.1 안정, "
        "0.1~0.25 중간 수준 변화, >0.25 유의미한 변화(재학습 검토 필요)\n"
    )
    lines.append("| target | period | psi_value | 해석 |")
    lines.append("|---|---|---|---|")
    for target_name, period, value in psi_rows:
        verdict = "안정" if value < 0.1 else ("중간 변화" if value < 0.25 else "유의미한 변화")
        lines.append(f"| {target_name} | {period} | {value:.4f} | {verdict} |")
    lines.append("")

    lines.append("## 7. SHAP 해석\n")
    if shap_image_path is not None:
        rel_path = Path(shap_image_path).resolve().relative_to(Path.cwd().resolve()).as_posix()
        lines.append(f"챔피언 모델(`{champion['champion']}`)의 SHAP summary plot (테스트셋 샘플 기준):\n")
        lines.append(f"![SHAP summary plot]({rel_path})\n")
    else:
        lines.append("SHAP 해석을 생성하지 못했습니다.\n")

    lines.append("## 8. 재현 방법\n")
    lines.append("```bash\npython run.py\npytest tests/\n```\n")

    lines.append("## 9. 참고\n")
    lines.append("- 상세 진행 로그: [PROGRESS.md](PROGRESS.md)")
    lines.append("- SQLite 스키마: [src/db.py](src/db.py)")

    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    print("[1/10] SQLite 스키마 초기화")
    init_db()
    conn = get_connection()

    print("[2/10] 원본 데이터 로드 및 전처리")
    raw_df = preprocess.load_raw()
    df = preprocess.basic_preprocess(raw_df)
    print(f"  -> {df.shape[0]:,} rows, {df.shape[1]} columns after preprocessing")

    print("[3/10] applicants 테이블 저장")
    preprocess.save_applicants(df, conn)

    print("[4/10] WoE/IV 계산 + 로지스틱 베이스라인 스코어카드 학습")
    baseline = train_baseline.run_baseline(df, conn)
    print(f"  -> baseline AUC={baseline['auc']:.4f} KS={baseline['ks']:.4f} Gini={baseline['gini']:.4f} "
          f"(features={len(baseline['selected_features'])})")

    print("[5/10] LightGBM 고도화 모델 학습")
    advanced = train_advanced.run_lightgbm(baseline["train_df"], baseline["test_df"], conn)
    print(f"  -> lightgbm AUC={advanced['auc']:.4f} KS={advanced['ks']:.4f} Gini={advanced['gini']:.4f}")

    print("[6/10] 챔피언-챌린저 판단 및 점수/등급 산출")
    champion = train_advanced.choose_champion(baseline, advanced)
    print(f"  -> champion={champion['champion']} ({champion['reason']})")

    if champion["champion"] == "lightgbm":
        champion_run_id = advanced["run_id"]
        full_proba = train_advanced.score_full_population(
            advanced["model"], df, advanced["feature_cols"], advanced["cat_cols"]
        ).to_numpy()
        ids_full = df[train_baseline.ID_COL].to_numpy()
        target_full = df[train_baseline.TARGET_COL].to_numpy()
    else:
        champion_run_id = baseline["run_id"]
        full_proba = baseline["proba_full"]
        ids_full = baseline["ids_full"]
        target_full = baseline["target_full"]

    full_score = scoring.probability_to_score(full_proba)
    score_series = pd.Series(full_score, index=df.index)
    grades = scoring.assign_grades(score_series)
    scoring.save_scores(ids_full, full_score, grades, champion_run_id, conn)

    grade_df = pd.DataFrame({"grade": grades.values, "score": full_score, "target": target_full})
    grade_stats = (
        grade_df.groupby("grade")
        .agg(count=("target", "size"), avg_score=("score", "mean"), default_rate=("target", "mean"))
        .reindex(scoring.GRADE_LABELS_ASCENDING[::-1])
        .reset_index()
    )

    print("[7/10] 컷오프 시뮬레이션")
    sim_df = simulation.simulate_cutoffs(score_series, pd.Series(target_full))
    simulation.save_cutoff_simulation(sim_df, champion_run_id, conn)

    print("[8/10] PSI 안정성 점검 (train vs test)")
    train_scores = score_series.loc[baseline["train_df"].index]
    test_scores = score_series.loc[baseline["test_df"].index]
    psi_rows = []
    score_psi = psi.calculate_psi(train_scores, test_scores)
    psi.save_psi(champion_run_id, "score", score_psi, "test_vs_train", conn)
    psi_rows.append(("score", "test_vs_train", score_psi))
    for feature in baseline["selected_features"][:5]:
        train_woe = wb.transform_feature_woe_values(baseline["train_df"][feature], baseline["fits"][feature])
        test_woe = wb.transform_feature_woe_values(baseline["test_df"][feature], baseline["fits"][feature])
        feature_psi = psi.calculate_psi(train_woe, test_woe)
        psi.save_psi(champion_run_id, feature, feature_psi, "test_vs_train", conn)
        psi_rows.append((feature, "test_vs_train", feature_psi))
    print(f"  -> score PSI={score_psi:.4f}")

    print("[9/10] SHAP 해석")
    try:
        if champion["champion"] == "lightgbm":
            X_shap = baseline["test_df"][advanced["feature_cols"]].copy()
            for c in advanced["cat_cols"]:
                X_shap[c] = X_shap[c].astype("category")
            X_shap = X_shap.sample(min(3000, len(X_shap)), random_state=42)
            shap_path = shap_explain.explain_champion(advanced["model"], X_shap, "lightgbm")
        else:
            X_shap = pd.DataFrame({
                c: wb.transform_feature_woe_values(baseline["test_df"][c], baseline["fits"][c])
                for c in baseline["selected_features"]
            })
            X_shap = X_shap.sample(min(3000, len(X_shap)), random_state=42)
            shap_path = shap_explain.explain_champion(baseline["model"], X_shap, "logistic")
        print(f"  -> {shap_path}")
    except Exception as e:
        print(f"  -> SHAP 생성 실패 (계속 진행): {e}")
        shap_path = None

    print("[10/10] REPORT.md 작성")
    report = build_report(
        baseline, advanced, champion, sim_df, grade_stats,
        n_rows=df.shape[0], n_features_selected=len(baseline["selected_features"]),
        psi_rows=psi_rows, shap_image_path=shap_path,
    )
    with open("REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    conn.close()
    update_progress_mvp_done()
    elapsed = time.time() - t0
    print(f"\n완료: {elapsed:.1f}s 소요. db/credit_scoring.db, REPORT.md 생성됨.")
    return 0


def update_progress_mvp_done() -> None:
    """Tick off the MVP checklist in PROGRESS.md now that run.py finished end-to-end."""
    try:
        with open("PROGRESS.md", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return

    mvp_start = content.find("## MVP 흐름")
    mvp_end = content.find("## 확장 흐름")
    if mvp_start == -1 or mvp_end == -1:
        return

    section = content[mvp_start:mvp_end]
    checked_section = section.replace("- [ ]", "- [x]")
    content = content[:mvp_start] + checked_section + content[mvp_end:]

    with open("PROGRESS.md", "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    sys.exit(main())
