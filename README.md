# credit-risk-scorecard

Home Credit Default Risk(Kaggle) 데이터를 사용해 실제 여신심사 업무 흐름을
처음부터 끝까지 재현하는 포트폴리오 프로젝트입니다.

변수처리(application + bureau + previous_application 결합 피처 엔지니어링) → WoE/IV
스코어카드 → 고도화 모델(LightGBM) → 챔피언-챌린저 비교 → 등급설계(A-E) →
컷오프 시뮬레이션(승인율/부도율/손실률) → 안정성(PSI)/해석(SHAP) →
Streamlit 대시보드 → 자소서/면접용 요약 문서(.docx)

## 챔피언-챌린저 결과 요약

application_train.csv 단독 대비, 신용정보(bureau) + 과거 대출 이력(previous_application)을
결합한 피처 엔지니어링으로 AUC가 로지스틱 +0.0059pt, LightGBM +0.0104pt 개선되었습니다.

| model | AUC | KS | Gini |
|---|---|---|---|
| 로지스틱 (WoE 스코어카드, 베이스라인) | 0.7487 | 0.3752 | 0.4973 |
| **LightGBM (고도화 모델, 챔피언)** | **0.7709** | **0.4057** | **0.5417** |

LightGBM이 베이스라인 대비 AUC +0.0222pt(사전 정의 채택 기준 0.01 초과)로 우세해
챔피언 모델로 채택했습니다. 판단 근거와 전체 수치는 [REPORT.md](REPORT.md)를 참고하세요.
새로 추가한 `BUREAU_DEBT_CREDIT_RATIO_MEAN`, `BUREAU_ACTIVE_RATIO`, `PREV_REFUSED_RATIO` 등이
IV/SHAP 상위권에 올라 실제로 예측력에 기여함을 확인했습니다.

등급별 실제 부도율(A가 가장 우량, E가 가장 위험):

| grade | count | avg_score | default_rate |
|---|---|---|---|
| A | 61,502 | 605.0 | 0.67% |
| B | 61,502 | 586.4 | 1.81% |
| C | 61,502 | 572.0 | 3.97% |
| D | 61,502 | 555.4 | 8.31% |
| E | 61,503 | 526.3 | 25.60% |

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

로컬 실행 시 `app.py`는 `db/credit_scoring.db`(run.py가 만든 전체 DB, 1GB+)를 그대로 읽습니다.

### 4-1. Streamlit Community Cloud 배포용 준비

전체 DB(`db/credit_scoring.db`)는 대부분 `features_woe`(768만 행)가 차지해 1GB가 넘어
GitHub/Streamlit Cloud에 올릴 수 없습니다. 대시보드가 실제로 쓰는 5개 테이블
(`model_runs`, `iv_summary`, `cutoff_simulation`, `psi_monitoring`, `scores`)만 담은
가벼운 `db/dashboard.db`(약 7MB)를 따로 만들어 커밋합니다. `app.py`는 `db/dashboard.db`가
있으면 그것을 우선 사용하고, 없으면 전체 DB로 자동 대체합니다.

```bash
python src/export_dashboard_db.py   # db/dashboard.db 생성 (git에 커밋 대상)
git add db/dashboard.db && git commit -m "Update dashboard.db" && git push
```

배포 시 Streamlit Cloud의 "Advanced settings → Python dependencies file"에
`requirements-app.txt`(streamlit/pandas/plotly만 포함, 빌드가 훨씬 빠름)를 지정하는 것을
권장합니다. 지정하지 않으면 기본 `requirements.txt`(lightgbm/shap 포함)로도 동작은 하지만
빌드 시간이 더 깁니다.

**Streamlit Cloud 배포 단계** (GitHub 로그인/OAuth 인증은 본인이 직접 진행해야 합니다):

1. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 계정으로 로그인
2. "Create app" → "Deploy a public app from GitHub" 선택
3. Repository: `Dajeong0315/credit-risk-scorecard`, Branch: `main`, Main file path: `app.py`
4. "Advanced settings"에서 Python dependencies file을 `requirements-app.txt`로 지정 (권장)
5. "Deploy" 클릭 → 빌드 완료 후 공개 URL 발급

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
  db/
    credit_scoring.db       # 전체 SQLite 결과 저장소, 1GB+ (git ignore)
    dashboard.db             # 대시보드/배포용 경량 DB, ~7MB (git 커밋 대상)
  outputs/
    shap_summary.png          # SHAP summary plot (git 커밋 대상, REPORT.md에서 참조)
    credit_risk_scorecard_summary.docx  # 자소서/면접 요약 문서 (git 커밋 대상)
    summary_data.json          # docx 생성용 중간 산출물 (git ignore)
  src/
    db.py                   # SQLite 스키마 초기화
    download_data.py        # Kaggle API 데이터 다운로드
    preprocess.py           # 결측치 처리 등 전처리
    feature_engineering.py  # bureau/previous_application 결합 피처 엔지니어링
    woe_binning.py          # WoE/IV 계산 및 저장
    train_baseline.py       # 로지스틱 스코어카드 베이스라인
    train_advanced.py       # LightGBM + 챔피언-챌린저 판단
    scoring.py               # 점수 스케일링(PDO) 및 A-E 등급
    simulation.py             # 컷오프 시뮬레이션
    psi.py                     # PSI 안정성 점검
    shap_explain.py            # SHAP 해석 (summary plot)
    generate_docx_data.py       # docx용 핵심 수치 JSON 추출
    export_dashboard_db.py       # 배포용 경량 DB(dashboard.db) 생성
  scripts/
    build_summary_docx.js       # 자소서/면접 요약 docx 생성 (docx-js)
  notebooks/                    # 탐색용 노트북 (src 모듈 재사용)
  app.py                        # Streamlit 대시보드
  run.py                        # MVP 전체 자동 실행 진입점
  requirements.txt              # 전체 의존성 (학습/분석 포함)
  requirements-app.txt          # app.py 전용 최소 의존성 (Streamlit Cloud 배포용)
  REPORT.md                     # 분석 리포트 (run.py가 자동 생성)
  PROGRESS.md                   # 단계별 진행 상황 기록
  tests/                        # pytest 단위테스트 (WoE/IV, 점수 스케일링, PSI, 피처 엔지니어링)
```

## 데이터베이스 스키마 (SQLite)

`applicants`, `features_woe`, `iv_summary`, `model_runs`, `scores`,
`cutoff_simulation`, `psi_monitoring` — 자세한 스키마는 [src/db.py](src/db.py) 참고.
모든 결과는 SQL로 직접 조회 가능합니다 (예: `SELECT * FROM cutoff_simulation WHERE run_id = 2`).

## 진행 상황

자세한 단계별 진행 상황과 의사결정 로그는 [PROGRESS.md](PROGRESS.md)를 참고하세요.
