# credit-risk-scorecard 분석 리포트

생성일: 2026-09-04

## 1. 데이터 개요

- 데이터: Home Credit Default Risk `application_train.csv`, 307,511건
- 전처리 후 사용 변수 수(ID/TARGET 제외): 156
- IV 기준(0.02~0.5) 선택된 변수 수: 25

## 2. IV(Information Value) 상위 변수

| feature_name | iv_value | bin_count |
|---|---|---|
| EXT_SOURCE_3 | 0.3298 | 11 |
| EXT_SOURCE_2 | 0.3032 | 11 |
| EXT_SOURCE_1 | 0.1513 | 11 |
| BUREAU_DEBT_CREDIT_RATIO_MEAN | 0.138 | 10 |
| BUREAU_DAYS_CREDIT_MEAN | 0.12 | 11 |
| DAYS_EMPLOYED | 0.1116 | 11 |
| AMT_GOODS_PRICE | 0.0914 | 11 |
| BUREAU_ACTIVE_RATIO | 0.0888 | 8 |
| DAYS_BIRTH | 0.0868 | 10 |
| OCCUPATION_TYPE | 0.0754 | 13 |
| BUREAU_DAYS_CREDIT_MIN | 0.0734 | 11 |
| PREV_REFUSED_RATIO | 0.0691 | 5 |
| CC_UTILIZATION_MEAN | 0.0657 | 9 |
| PREV_APPROVED_RATIO | 0.061 | 7 |
| CC_CNT_DRAWINGS_CURRENT_MEAN | 0.0609 | 8 |

## 3. 모델 성능 비교 (챔피언-챌린저)

| model | AUC | KS | Gini |
|---|---|---|---|
| 로지스틱 (WoE 스코어카드, 베이스라인) | 0.7530 | 0.3805 | 0.5061 |
| LightGBM (고도화 모델) | 0.7821 | 0.4281 | 0.5643 |

**챔피언 모델: `lightgbm`**

- AUC 개선폭: +0.0291
- 판단 사유: LightGBM이 베이스라인 대비 AUC +0.0291pt 우세 (기준 0.01 초과) -> 챔피언으로 채택

## 4. 점수 스케일링 및 등급 설계

- PDO(Points to Double Odds) 방식: base_score=600, base_odds=50:1, pdo=20
- 등급: A(최우량, 저위험) ~ E(최고위험), 동일 인원수(quantile) 기준 5등급

| grade | count | avg_score | default_rate |
|---|---|---|---|
| A | 61502 | 606.9 | 0.0064 |
| B | 61502 | 587.9 | 0.0170 |
| C | 61502 | 573.4 | 0.0356 |
| D | 61502 | 556.3 | 0.0794 |
| E | 61503 | 525.3 | 0.2651 |

## 5. 컷오프 시뮬레이션

- 가정: LGD(Loss Given Default) = 45% (Basel 무담보 소매여신 표준 가정)

| cutoff score | approval_rate | expected_default_rate | expected_loss |
|---|---|---|---|
| 516.1 | 95.00% | 6.02% | 2.71% |
| 529.5 | 90.00% | 4.89% | 2.20% |
| 538.5 | 85.00% | 4.09% | 1.84% |
| 545.4 | 80.00% | 3.46% | 1.56% |
| 551.4 | 75.00% | 2.98% | 1.34% |
| 556.6 | 70.00% | 2.58% | 1.16% |
| 561.3 | 65.00% | 2.26% | 1.02% |
| 565.6 | 60.00% | 1.97% | 0.89% |
| 569.7 | 55.00% | 1.74% | 0.78% |
| 573.4 | 50.00% | 1.53% | 0.69% |
| 577.1 | 45.00% | 1.34% | 0.60% |
| 580.7 | 40.00% | 1.17% | 0.53% |
| 584.3 | 35.00% | 1.01% | 0.45% |
| 587.8 | 30.00% | 0.90% | 0.40% |
| 591.6 | 25.00% | 0.76% | 0.34% |
| 595.5 | 20.00% | 0.64% | 0.29% |
| 599.9 | 15.00% | 0.50% | 0.23% |
| 605.0 | 10.00% | 0.39% | 0.17% |
| 612.1 | 5.00% | 0.31% | 0.14% |

## 6. PSI(Population Stability Index) 안정성 점검

- 기준(expected)=train, 비교 대상(actual)=test 분포. 해석: PSI<0.1 안정, 0.1~0.25 중간 수준 변화, >0.25 유의미한 변화(재학습 검토 필요)

| target | period | psi_value | 해석 |
|---|---|---|---|
| score | test_vs_train | 0.0002 | 안정 |
| EXT_SOURCE_3 | test_vs_train | 0.0001 | 안정 |
| EXT_SOURCE_2 | test_vs_train | 0.0001 | 안정 |
| EXT_SOURCE_1 | test_vs_train | 0.0002 | 안정 |
| BUREAU_DEBT_CREDIT_RATIO_MEAN | test_vs_train | 0.0001 | 안정 |
| BUREAU_DAYS_CREDIT_MEAN | test_vs_train | 0.0001 | 안정 |

## 7. SHAP 해석

챔피언 모델(`lightgbm`)의 SHAP summary plot (테스트셋 샘플 기준):

![SHAP summary plot](outputs/shap_summary.png)

## 8. 재현 방법

```bash
python run.py
pytest tests/
```

## 9. 참고

- 상세 진행 로그: [PROGRESS.md](PROGRESS.md)
- SQLite 스키마: [src/db.py](src/db.py)