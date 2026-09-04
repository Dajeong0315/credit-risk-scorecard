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
- [ ] (선택, 사용자 확인 후) 배포 — 아직 요청 없음, 보류

## 결정 / 이슈 로그

- 2026-09-04: 데이터 다운로드를 위해 Kaggle API 키 필요. 사용자가 Kaggle의 신규 API 토큰(`KGAT_...`, kaggle.json이 아닌 `~/.kaggle/access_token` 방식)을 발급 → `~/.kaggle/access_token`에 저장, 대회 규칙(Join Competition) 동의 후 다운로드 성공.
- 2026-09-04: pandas 3.0.3의 `future.infer_string` 기본값 때문에 문자열 컬럼 dtype이 `object`가 아닌 `str`로 표시됨. LightGBM 카테고리 컬럼 감지 로직(`dtype == "object"`)이 이를 놓쳐 학습이 실패 → `not pd.api.types.is_numeric_dtype(...)` 기준으로 수정.
- 2026-09-04: `preprocess.save_applicants`가 `df.to_dict(orient="records")`를 사용해 30만 행 x 105열 처리가 매우 느렸음(수 분 이상) → `.values.tolist()` 기반으로 재작성해 속도 개선(`run.py` 전체 실행 시간 약 180초).
- 2026-09-04: 챔피언-챌린저 결과 — LightGBM(AUC 0.7605)이 로지스틱 WoE 베이스라인(AUC 0.7428) 대비 +0.0177pt 우세, 채택 기준(0.01) 초과 → LightGBM을 챔피언으로 채택. 근거는 REPORT.md 3절 참고.
- 2026-09-04: IV 기준 변수 선택 임계값은 0.02~0.5(스코어카드 업계 표준: <0.02 무의미, >0.5 의심스러운 leakage) 사용, 상위 25개 변수로 로지스틱 스코어카드 구성.
- 2026-09-04: PSI는 train vs test 랜덤 분할 기준으로 계산(스펙에서 "train vs test 또는 time-split" 중 선택 가능). 결과 PSI≈0.0003으로 매우 안정적 — 랜덤 분할이라 당연한 결과이므로, 실제 운영 환경이라면 time-split 기준 PSI를 추가로 보는 것을 권장한다는 점을 REPORT.md에 남기지는 않았으나 참고 사항으로 기록.
- 2026-09-04: docx 생성용 `docx` npm 패키지가 이 로컬 환경에는 사전 설치되어 있지 않아 프로젝트 폴더에 로컬 설치(`npm install docx`, node_modules는 git ignore)로 진행. 스킬이 제시한 LibreOffice 기반 PDF 변환 검증 스크립트(soffice.py)는 Windows에서 미지원(AF_UNIX 소켓 의존)이라 실행 불가 — 대신 docx(zip) 구조와 XML 텍스트 추출로 내용을 검증함.
- 2026-09-04: GitHub 저장소(https://github.com/Dajeong0315/credit-risk-scorecard, public)에 연결 완료. 커밋 author는 GitHub noreply 이메일(Dajeong0315@users.noreply.github.com) 사용 — 공개 저장소라 실제 이메일 노출을 피하기 위함(사용자 확인 후 결정).
