# credit-risk-scorecard

Home Credit Default Risk(Kaggle) 데이터를 사용해 실제 여신심사 업무 흐름을
처음부터 끝까지 재현하는 포트폴리오 프로젝트입니다.

변수처리 → WoE/IV 스코어카드 → 고도화 모델(LightGBM) → 챔피언-챌린저 비교 →
등급설계(A-E) → 컷오프 시뮬레이션(승인율/부도율/손실률) → 안정성(PSI)/해석(SHAP)
→ Streamlit 대시보드 → 자소서/면접용 요약 문서(.docx)

## 챔피언-챌린저 결과 요약

| model | AUC | KS | Gini |
|---|---|---|---|
| 로지스틱 (WoE 스코어카드, 베이스라인) | 0.7428 | 0.3684 | 0.4856 |
| **LightGBM (고도화 모델, 챔피언)** | **0.7605** | **0.3922** | **0.5209** |

LightGBM이 베이스라인 대비 AUC +0.0177pt(사전 정의 채택 기준 0.01 초과)로 우세해
챔피언 모델로 채택했습니다. 판단 근거와 전체 수치는 [REPORT.md](REPORT.md)를 참고하세요.

등급별 실제 부도율(A가 가장 우량, E가 가장 위험):

| grade | count | avg_score | default_rate |
|---|---|---|---|
| A | 61,502 | 602.0 | 0.87% |
| B | 61,502 | 584.4 | 2.26% |
| C | 61,502 | 570.6 | 4.52% |
| D | 61,502 | 554.8 | 8.86% |
| E | 61,503 | 527.4 | 23.86% |

## 실행 방법

### 0. 의존성 설치

```bash
pip install -r requirements.txt
```

### 1. 데이터 준비 (Kaggle API)

[Home Credit Default Risk 대회](https://www.kaggle.com/competitions/home-credit-default-risk) 페이지에서
**Join Competition**(규칙 동의)을 먼저 진행해야 API 다운로드가 허용됩니다.

Kaggle 계정 설정(kaggle.com/settings → API)에서 토큰을 발급받아 아래 중 하나로 저장하세요:

- 신규 방식(권장): `~/.kaggle/access_token` 파일에 토큰 문자열만 저장
- 기존 방식: `~/.kaggle/kaggle.json` 에 `{"username": "...", "key": "..."}` 저장

```bash
python src/download_data.py
```

### 2. MVP 파이프라인 실행

전처리 → WoE/IV → 로지스틱 베이스라인 → LightGBM → 챔피언-챌린저 → 등급/컷오프
시뮬레이션 → PSI → SHAP → REPORT.md 생성까지 한 번에 자동 실행됩니다.

```bash
python run.py
```

`db/credit_scoring.db`(SQLite), `REPORT.md`, `outputs/shap_summary.png` 가 생성됩니다.
(로컬 환경 기준 약 2~3분 소요)

### 3. 테스트

```bash
pytest tests/
```

### 4. Streamlit 대시보드 (로컬 전용)

```bash
streamlit run app.py
```

모델 실행(run) 선택, 컷오프 슬라이더로 승인율/예상 부도율/예상 손실률을 실시간으로 확인할 수 있습니다.

### 5. 자소서/면접용 요약 문서(.docx) 생성

SQLite 결과에서 핵심 수치를 자동 추출해 Word 문서를 생성합니다 (Node.js 필요).

```bash
npm install docx
python src/generate_docx_data.py
node scripts/build_summary_docx.js
```

`outputs/credit_risk_scorecard_summary.docx` 가 생성됩니다.

## 프로젝트 구조

```
credit-risk-scorecard/
  data/raw/                 # 원본 데이터 (git ignore)
  db/credit_scoring.db      # SQLite 결과 저장소 (git ignore)
  outputs/                  # SHAP 이미지, docx 등 생성 산출물 (git ignore)
  src/
    db.py                   # SQLite 스키마 초기화
    download_data.py        # Kaggle API 데이터 다운로드
    preprocess.py           # 결측치 처리 등 전처리
    woe_binning.py          # WoE/IV 계산 및 저장
    train_baseline.py       # 로지스틱 스코어카드 베이스라인
    train_advanced.py       # LightGBM + 챔피언-챌린저 판단
    scoring.py               # 점수 스케일링(PDO) 및 A-E 등급
    simulation.py             # 컷오프 시뮬레이션
    psi.py                     # PSI 안정성 점검
    shap_explain.py            # SHAP 해석 (summary plot)
    generate_docx_data.py       # docx용 핵심 수치 JSON 추출
  scripts/
    build_summary_docx.js       # 자소서/면접 요약 docx 생성 (docx-js)
  notebooks/                    # 탐색용 노트북 (src 모듈 재사용)
  app.py                        # Streamlit 대시보드
  run.py                        # MVP 전체 자동 실행 진입점
  REPORT.md                     # 분석 리포트 (run.py가 자동 생성)
  PROGRESS.md                   # 단계별 진행 상황 기록
  tests/                        # pytest 단위테스트 (WoE/IV, 점수 스케일링, PSI)
```

## 데이터베이스 스키마 (SQLite)

`applicants`, `features_woe`, `iv_summary`, `model_runs`, `scores`,
`cutoff_simulation`, `psi_monitoring` — 자세한 스키마는 [src/db.py](src/db.py) 참고.
모든 결과는 SQL로 직접 조회 가능합니다 (예: `SELECT * FROM cutoff_simulation WHERE run_id = 2`).

## 진행 상황

자세한 단계별 진행 상황과 의사결정 로그는 [PROGRESS.md](PROGRESS.md)를 참고하세요.
