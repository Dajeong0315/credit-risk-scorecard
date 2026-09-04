# credit-risk-scorecard

Home Credit Default Risk(Kaggle) 데이터를 사용해 실제 여신심사 업무 흐름을
처음부터 끝까지 재현하는 포트폴리오 프로젝트.

변수처리 → WoE/IV 스코어카드 → 고도화 모델(LightGBM) → 챔피언-챌린저 비교 →
등급설계(A-E) → 컷오프 시뮬레이션(승인율/부도율/손실률) → 안정성(PSI)/해석(SHAP)

## 실행 방법

### 1. 데이터 준비

Kaggle 계정에서 API 토큰을 발급받아 `~/.kaggle/kaggle.json`에 저장하고,
[Home Credit Default Risk 대회](https://www.kaggle.com/competitions/home-credit-default-risk)에서
Join Competition(규칙 동의)을 먼저 진행해야 합니다.

```bash
pip install -r requirements.txt
python src/download_data.py
```

### 2. 전체 파이프라인 실행 (MVP)

```bash
python run.py
```

`db/credit_scoring.db`(SQLite)와 `REPORT.md`가 생성됩니다.

### 3. 테스트

```bash
pytest tests/
```

## 프로젝트 구조

```
credit-risk-scorecard/
  data/raw/            # 원본 데이터 (git ignore)
  db/credit_scoring.db # SQLite 결과 저장소
  src/
    db.py              # SQLite 스키마 초기화
    download_data.py   # Kaggle API 데이터 다운로드
    preprocess.py       # 결측치 처리 등 전처리
    woe_binning.py      # WoE/IV 계산 및 저장
    train_baseline.py   # 로지스틱 스코어카드
    train_advanced.py   # LightGBM + 챔피언-챌린저
    scoring.py           # 점수 스케일링 및 A-E 등급
    simulation.py         # 컷오프 시뮬레이션
    psi.py                 # PSI 안정성 점검 (확장)
    shap_explain.py        # SHAP 해석 (확장)
  notebooks/            # 탐색용 노트북 (src 모듈 재사용)
  app.py                 # Streamlit 대시보드 (확장)
  run.py                 # MVP 전체 자동 실행 진입점
  REPORT.md               # 분석 리포트 (자동 생성)
  PROGRESS.md             # 진행 상황 기록
  tests/
```

## 데이터베이스 스키마 (SQLite)

`applicants`, `features_woe`, `iv_summary`, `model_runs`, `scores`,
`cutoff_simulation`, `psi_monitoring` — 자세한 스키마는 [src/db.py](src/db.py) 참고.

## 진행 상황

자세한 단계별 진행 상황은 [PROGRESS.md](PROGRESS.md)를 참고하세요.
