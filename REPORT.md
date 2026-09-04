# credit-risk-scorecard 분석 리포트

생성일: 2026-09-04

## 1. 데이터 개요

- 데이터: Home Credit Default Risk `application_train.csv`, 307,511건
- 전처리 후 사용 변수 수(ID/TARGET 제외): 103
- IV 기준(0.02~0.5) 선택된 변수 수: 25

## 2. IV(Information Value) 상위 변수

| feature_name | iv_value | bin_count |
|---|---|---|
| EXT_SOURCE_3 | 0.3298 | 11 |
| EXT_SOURCE_2 | 0.3032 | 11 |
| EXT_SOURCE_1 | 0.1513 | 11 |
| DAYS_EMPLOYED | 0.1116 | 11 |
| AMT_GOODS_PRICE | 0.0914 | 11 |
| DAYS_BIRTH | 0.0868 | 10 |
| OCCUPATION_TYPE | 0.0754 | 13 |
| ORGANIZATION_TYPE | 0.0596 | 17 |
| NAME_INCOME_TYPE | 0.0559 | 5 |
| REGION_RATING_CLIENT_W_CITY | 0.0516 | 3 |
| NAME_EDUCATION_TYPE | 0.0493 | 5 |
| REGION_RATING_CLIENT | 0.0482 | 3 |
| AMT_CREDIT | 0.0448 | 10 |
| DAYS_LAST_PHONE_CHANGE | 0.0446 | 10 |
| CODE_GENDER | 0.0385 | 3 |

## 3. 모델 성능 비교 (챔피언-챌린저)

| model | AUC | KS | Gini |
|---|---|---|---|
| 로지스틱 (WoE 스코어카드, 베이스라인) | 0.7428 | 0.3684 | 0.4856 |
| LightGBM (고도화 모델) | 0.7605 | 0.3922 | 0.5209 |

**챔피언 모델: `lightgbm`**

- AUC 개선폭: +0.0177
- 판단 사유: LightGBM이 베이스라인 대비 AUC +0.0177pt 우세 (기준 0.01 초과) -> 챔피언으로 채택

## 4. 점수 스케일링 및 등급 설계

- PDO(Points to Double Odds) 방식: base_score=600, base_odds=50:1, pdo=20
- 등급: A(최우량, 저위험) ~ E(최고위험), 동일 인원수(quantile) 기준 5등급

| grade | count | avg_score | default_rate |
|---|---|---|---|
| A | 61502 | 602.0 | 0.0087 |
| B | 61502 | 584.4 | 0.0226 |
| C | 61502 | 570.6 | 0.0452 |
| D | 61502 | 554.8 | 0.0886 |
| E | 61503 | 527.4 | 0.2386 |

## 5. 컷오프 시뮬레이션

- 가정: LGD(Loss Given Default) = 45% (Basel 무담보 소매여신 표준 가정)

| cutoff score | approval_rate | expected_default_rate | expected_loss |
|---|---|---|---|
| 519.4 | 95.00% | 6.39% | 2.88% |
| 530.8 | 90.00% | 5.42% | 2.44% |
| 538.7 | 85.00% | 4.68% | 2.11% |
| 545.0 | 80.00% | 4.13% | 1.86% |
| 550.4 | 75.00% | 3.63% | 1.63% |
| 555.1 | 70.00% | 3.22% | 1.45% |
| 559.4 | 65.00% | 2.85% | 1.28% |
| 563.4 | 60.00% | 2.55% | 1.15% |
| 567.1 | 55.00% | 2.29% | 1.03% |
| 570.6 | 50.00% | 2.04% | 0.92% |
| 574.1 | 45.00% | 1.81% | 0.82% |
| 577.5 | 40.00% | 1.57% | 0.70% |
| 580.9 | 35.00% | 1.37% | 0.61% |
| 584.3 | 30.00% | 1.20% | 0.54% |
| 587.8 | 25.00% | 1.03% | 0.47% |
| 591.6 | 20.00% | 0.87% | 0.39% |
| 595.6 | 15.00% | 0.69% | 0.31% |
| 600.3 | 10.00% | 0.55% | 0.25% |
| 606.7 | 5.00% | 0.34% | 0.15% |

## 6. 재현 방법

```bash
python run.py
pytest tests/
```

## 7. 참고

- 상세 진행 로그: [PROGRESS.md](PROGRESS.md)
- SQLite 스키마: [src/db.py](src/db.py)