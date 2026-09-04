# PROGRESS

작업 중 컨텍스트가 끊겨도 이어갈 수 있도록 단계별 진행 상황을 기록한다.
막힌 부분/임의 판단은 이유와 함께 남긴다.

## MVP 흐름

- [x] 프로젝트 골격 구성 (폴더 구조, requirements.txt, .gitignore, PROGRESS.md)
- [x] 데이터 다운로드 (Kaggle API, `home-credit-default-risk`, application_train.csv 307,511행)
- [x] 데이터 적재 → SQLite `applicants` 저장
- [x] 전처리 (결측치 처리 등)
- [x] WoE/IV binning → `features_woe`, `iv_summary` 저장, IV 기준 변수 선택
- [x] 로지스틱회귀 스코어카드 베이스라인 학습 → `model_runs`, `scores` 저장
- [x] LightGBM 학습 → AUC/KS/Gini 비교
- [x] 챔피언-챌린저 판단 (베이스라인 vs 고도화 모델)
- [x] 점수 스케일링 및 A-E 등급 설계
- [x] 컷오프별 시뮬레이션 → `cutoff_simulation` 저장
- [x] REPORT.md 작성 (성능 비교표 + 컷오프 시뮬레이션 표)
- [x] pytest 단위테스트 (WoE/IV, 점수 스케일링)
- [x] `python run.py` 전체 자동 실행 확인

## 확장 흐름 (MVP 완료 후)

- [x] PSI 안정성 점검 (`psi_monitoring`, train vs test, score + top 5 IV 변수)
- [x] SHAP 해석 (summary plot → REPORT.md 삽입, `outputs/shap_summary.png`)
- [x] Streamlit 대시보드 (`app.py`, 컷오프 슬라이더 인터랙티브 확인 완료)
- [x] 자소서/면접용 요약 문서 (.docx, DB에서 핵심 수치 자동 추출)
- [x] 챔피언-챌린저 비교 섹션 정리 (REPORT.md 3절, README.md 요약 섹션)
- [x] (선택, 사용자 확인 후) 배포 — 사용자가 Streamlit Cloud 배포 요청. 저장소/DB/requirements 준비 완료, 실제 배포(GitHub OAuth 로그인 포함)는 사용자가 share.streamlit.io에서 직접 진행해야 함 (README "4-1. Streamlit Community Cloud 배포용 준비" 참고)

## 스펙 외 추가 고도화 (사용자 요청, 2026-09-04)

- [x] bureau.csv + bureau_balance.csv + previous_application.csv 결합 피처 엔지니어링 (`src/feature_engineering.py`)
  - AUC 개선: 로지스틱 0.7428→0.7487(+0.0059), LightGBM 0.7605→0.7709(+0.0104)
  - 신규 피처(BUREAU_DEBT_CREDIT_RATIO_MEAN, BUREAU_ACTIVE_RATIO, PREV_REFUSED_RATIO 등)가 IV/SHAP 상위권에 실제로 등장해 기여도 확인
- [x] POS_CASH_balance.csv + credit_card_balance.csv + installments_payments.csv 결합 (SK_ID_CURR을 직접 보유하고 있어 2단계 롤업 불필요, 예상보다 단순했음)
  - AUC 추가 개선: 로지스틱 0.7487→0.7530(+0.0043), LightGBM 0.7709→0.7821(+0.0112)
  - **application_train.csv 단독 대비 누적 개선**: 로지스틱 0.7428→0.7530(+0.0102), LightGBM 0.7605→0.7821(+0.0216)
  - `INSTAL_DPD_MEAN`(할부상환 연체일수)이 SHAP 중요도 전체 4위(EXT_SOURCE 3개 다음)로 등극 — 가장 효과적인 신규 피처
  - `CC_UTILIZATION_MEAN`(신용카드 이용률)도 IV 상위 15위 안에 진입

## 결정 / 이슈 로그

- 2026-09-04: 데이터 다운로드를 위해 Kaggle API 키 필요. 사용자가 Kaggle의 신규 API 토큰(`KGAT_...`, kaggle.json이 아닌 `~/.kaggle/access_token` 방식)을 발급 → `~/.kaggle/access_token`에 저장, 대회 규칙(Join Competition) 동의 후 다운로드 성공.
- 2026-09-04: pandas 3.0.3의 `future.infer_string` 기본값 때문에 문자열 컬럼 dtype이 `object`가 아닌 `str`로 표시됨. LightGBM 카테고리 컬럼 감지 로직(`dtype == "object"`)이 이를 놓쳐 학습이 실패 → `not pd.api.types.is_numeric_dtype(...)` 기준으로 수정.
- 2026-09-04: `preprocess.save_applicants`가 `df.to_dict(orient="records")`를 사용해 30만 행 x 105열 처리가 매우 느렸음(수 분 이상) → `.values.tolist()` 기반으로 재작성해 속도 개선(`run.py` 전체 실행 시간 약 180초).
- 2026-09-04: 챔피언-챌린저 결과 — LightGBM(AUC 0.7605)이 로지스틱 WoE 베이스라인(AUC 0.7428) 대비 +0.0177pt 우세, 채택 기준(0.01) 초과 → LightGBM을 챔피언으로 채택. 근거는 REPORT.md 3절 참고.
- 2026-09-04: IV 기준 변수 선택 임계값은 0.02~0.5(스코어카드 업계 표준: <0.02 무의미, >0.5 의심스러운 leakage) 사용, 상위 25개 변수로 로지스틱 스코어카드 구성.
- 2026-09-04: PSI는 train vs test 랜덤 분할 기준으로 계산(스펙에서 "train vs test 또는 time-split" 중 선택 가능). 결과 PSI≈0.0003으로 매우 안정적 — 랜덤 분할이라 당연한 결과이므로, 실제 운영 환경이라면 time-split 기준 PSI를 추가로 보는 것을 권장한다는 점을 REPORT.md에 남기지는 않았으나 참고 사항으로 기록.
- 2026-09-04: docx 생성용 `docx` npm 패키지가 이 로컬 환경에는 사전 설치되어 있지 않아 프로젝트 폴더에 로컬 설치(`npm install docx`, node_modules는 git ignore)로 진행. 스킬이 제시한 LibreOffice 기반 PDF 변환 검증 스크립트(soffice.py)는 Windows에서 미지원(AF_UNIX 소켓 의존)이라 실행 불가 — 대신 docx(zip) 구조와 XML 텍스트 추출로 내용을 검증함.
- 2026-09-04: GitHub 저장소(https://github.com/Dajeong0315/credit-risk-scorecard, public)에 연결 완료. 커밋 author는 GitHub noreply 이메일(Dajeong0315@users.noreply.github.com) 사용 — 공개 저장소라 실제 이메일 노출을 피하기 위함(사용자 확인 후 결정).
- 2026-09-04: Streamlit Cloud 배포 준비. 전체 db(1.6GB, 대부분 features_woe 768만 행)는 GitHub/Cloud에 올릴 수 없어, app.py가 실제로 쓰는 5개 테이블만 담은 `db/dashboard.db`(~7MB)를 별도로 export해 커밋(`src/export_dashboard_db.py`). app.py는 dashboard.db가 있으면 우선 사용, 없으면 전체 db로 자동 폴백하도록 수정. Streamlit Cloud용 최소 requirements-app.txt(streamlit/pandas/plotly만)도 별도 추가 — 빌드 시간 단축 목적. 실제 배포(GitHub 계정 로그인/OAuth 인증)는 Claude가 대신할 수 없는 영역이라 사용자가 share.streamlit.io에서 직접 진행해야 함을 안내.
- 2026-09-04: 배포 직후 사용자가 "글씨가 회색이라 잘 안 보임 / 차트 크기 조정 필요" 피드백. 원인: `.streamlit/config.toml`로 테마를 고정하지 않아 뷰어의 시스템 다크/라이트 설정에 따라 렌더링이 달라졌고, Plotly 차트는 라이트 배경 기준 색상을 하드코딩해둬서 불일치가 있었던 것으로 추정 → 라이트 테마를 명시적으로 고정하고, 차트 폰트/눈금/범례 색상과 높이를 명시적으로 키움. 로컬에서 라이트 테마로 재확인 후 배포, 이후 사용자가 정상 확인.
- 2026-09-04: 사용자가 스펙에 없던 추가 고도화(다른 테이블 결합 피처 엔지니어링)를 요청 → bureau/bureau_balance/previous_application을 SK_ID_CURR 단위로 집계해 병합하는 `src/feature_engineering.py` 추가. run.py 파이프라인에 전처리 직후 단계로 통합(총 11단계로 변경). AUC 개선 확인(로지스틱 +0.0059, LightGBM +0.0104) → REPORT.md/README.md 수치 갱신, dashboard.db/docx 재생성 후 커밋.
- 2026-09-04: 사용자가 이어서 POS_CASH_balance/credit_card_balance/installments_payments도 결합 요청. 세 테이블 모두 SK_ID_CURR을 직접 보유해 SK_ID_PREV 경유 2단계 롤업 없이 바로 집계 가능 — 애초 우려했던 것보다 단순했음. installments_payments.csv(723MB)가 가장 큰 원본 파일이라 전체 파이프라인 실행 시간이 ~180초→~226초로 늘어남. 결과: LightGBM AUC 0.7709→0.7821(+0.0112 추가, application_train 단독 대비 총 +0.0216). `INSTAL_DPD_MEAN`이 SHAP 전체 4위로 EXT_SOURCE 3개 다음으로 중요한 변수가 됨 — 상환 연체 패턴이 매우 강한 신호임을 확인. 새 집계 함수 3개에 대한 pytest 6개 추가(총 21개 통과).
