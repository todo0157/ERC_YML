## 학부연구 보고서(최종)

- **연구 주제**: 압출 기반 적층제조(Extrusion-based AM) 공정에서 **PM/VOC 시계열 데이터를 이용한 표면 거칠기(또는 거칠기와 연관된 품질 지표) 예측** 및 중요 특징(Feature) 분석
- **작성 범위**: `YML/ERC/` 폴더 내 코드·데이터·결과물(엑셀/이미지) 전수 검토 기반  
  - 최종 결과는 **`ERC/20260115_3,5,10/`(실제 Roughness 타겟, p10~p100 포함)** 실험을 기준으로 정리함
- **작성 언어/분량**: 한국어, A4 10장 이상(보고서 형식)
- **작성일**: 2026-01-15
- **작성자**: (기입)

---

## 초록(Abstract)

본 연구는 압출 기반 적층제조 공정에서 공정 진행 중 취득되는 **입자(PM) 및 휘발성 유기화합물(VOC) 시계열**로부터 자동 특징추출(`tsfresh`)과 특징공학(지연/이동평균/상호작용)을 수행하고, 이를 기반으로 공정 품질 지표(표면 거칠기 또는 거칠기와 강하게 연관된 목표값)를 예측하는 모델을 구축·분석하였다.  
연구의 핵심은 (1) 공정의 부분 구간(p10~p100)만으로도 전체 품질을 예측하기 위해 구간별 특징을 추출하고, (2) **VOC의 지연(Lag) 및 누적(Rolling) 효과**를 반영한 파생 특징을 설계하며, (3) PM과 VOC의 곱(Product) 형태 상호작용 특징을 생성하여 통합(Integrated) 데이터셋의 예측력을 평가하는 것이다. 또한 (4) RBF-SVR처럼 모델 자체의 중요도 산출이 어려운 경우를 위해 **Permutation Importance 기반의 특징 랭킹**을 적용하고, (5) 자동 생성된 상호작용 특징에 “실제로는 단일 센서의 상수/완전상관 성분이 섞여 상호작용처럼 보이는” 문제가 존재할 수 있음을 확인하여 **상호작용 특징 감사(Audit) 및 재분류 절차**를 설계·적용하였다.

실험은 크게 (A) 기존 실험/분석(`ERC/05_Comparative_Analysis_PM`, `ERC/07_Comparative_Analysis_Roughness`, `ERC/code/Model/*`)과, (B) 시간 간격(지연/윈도우)을 변경한 최신 실험 파이프라인으로 구성된다. 본 최종 보고서에서는 **`ERC/20260115_3,5,10`(lag/window=[3,5,10], p10~p100 포함)** 실험을 중심으로 결과를 제시한다. 이 실험은 `Printing_qualitydata.xlsx`의 **실제 `Roughness(nm)`**를 타겟으로 병합하여 수행되었으며, Integrated 데이터셋 기준 평균 상대오차는 **ExtraTrees 7.20%**, **SVR 7.44%**로 나타났다. 또한 상호작용 특징 전수 감사 결과, 상호작용으로 분류된 항목 중 **346개가 재분류 대상으로 식별**되었고, 이 중 **306개는 PM 단독(PM_Raw)**, **40개는 VOC 단독(VOC_Enhanced)**으로 분류되어 분석 타당성을 확보하였다.

---

## 목차

- **1. 서론**
  - 1.1 연구 배경 및 문제 정의
  - 1.2 연구 목표와 기여
  - 1.3 보고서 구성
- **2. 데이터 및 전처리**
  - 2.1 데이터 구성 및 파일 구조
  - 2.2 공정 구간(p10~p100) 정의
  - 2.3 결측치 처리 및 주의점(bfill 등)
  - 2.4 타겟(Target) 정의 및 정합성 확보(실제 Roughness 병합)
- **3. 특징 추출 및 특징공학**
  - 3.1 `tsfresh` 자동 특징추출 개요
  - 3.2 VOC 파생 특징: Lag / Rolling / Rolling+Lag
  - 3.3 PM–VOC 상호작용 특징: Product 기반 통합 특징
  - 3.4 특징 타입(Type) 분류 체계 및 라벨링
- **4. 모델링 및 평가 방법**
  - 4.1 ExtraTrees 회귀(MDI 중요도)
  - 4.2 SVR(RBF) + Target Scaling
  - 4.3 Permutation Importance 기반 SVR 특징 랭킹
  - 4.4 평가 지표: RE(Relative Error) 정의 및 단위(비율 vs %)
  - 4.5 Best-K 선택(구간별 K 최적화)
- **5. 실험 설계**
  - 5.1 기존 파이프라인(Enhanced/TSFRESH/거칠기 타겟 결합)
  - 5.2 최신 실험 파이프라인(20260115_3,5,10: 실제 Roughness 타겟)
  - 5.3 비교 케이스: PM_Only / VOC_Only / Integrated
- **6. 결과**
  - 6.1 Best-K 성능 비교(그래프 및 요약표)
  - 6.2 특징 타입 분포(Best-K 기준)
  - 6.3 공통 Top 특징(모델 교집합) 및 유사도(Jaccard)
  - 6.4 구간 간 지속성(빈도) 분석
- **7. 상호작용 특징 감사(Audit) 및 재분류**
  - 7.1 “가짜 상호작용” 발생 메커니즘(상수/완전상관)
  - 7.2 전수 감사 절차 및 기준(상관계수 0.9999)
  - 7.3 재분류 규칙(A안 포함) 및 결과 반영(엑셀/그래프)
- **8. 논의**
  - 8.1 지연/누적 효과의 해석: 왜 과거 VOC가 도움이 될 수 있는가
  - 8.2 모델별 성능 차이(ExtraTrees vs SVR) 해석
  - 8.3 연구 타당성/재현성 관점의 점검(라벨, 누수, 단위)
- **9. 결론 및 향후 연구**
- **참고문헌**
- **부록**

---

## 1. 서론

### 1.1 연구 배경 및 문제 정의

압출 기반 적층제조 공정에서는 노즐, 소재, 환경 조건, 배출 상태에 따라 표면 품질이 달라질 수 있으며, 그 결과로 **표면 거칠기(Roughness)** 같은 품질 지표가 변동한다. 공정 중 실시간으로 취득 가능한 센서 데이터(예: **PM(Particle Matter) 계열**과 **VOC(Volatile Organic Compounds)** 계열)는 재료의 상태 변화, 배출 과정의 불안정성, 혹은 반응(응축/입자화 등)의 간접적 신호를 제공할 수 있다.  
본 연구의 문제 정의는 다음과 같다.

- **문제 1**: 공정이 끝나기 전(p10~p90 등)의 일부 시계열만으로도 최종 품질(거칠기 또는 이에 준하는 타겟)을 예측할 수 있는가?
- **문제 2**: VOC는 즉시 효과보다 **시간 지연(Lag)** 또는 **누적(Rolling)** 형태로 품질에 반영될 가능성이 있는데, 이를 특징으로 설계했을 때 예측력이 향상되는가?
- **문제 3**: PM과 VOC를 통합한 **상호작용(Interaction)** 특징이 단일 센서 기반 특징보다 효과적인가?
- **문제 4**: 자동 생성된 수많은 특징 중 “정말로 상호작용을 의미하는 특징”만을 선별하고, 분석의 타당성을 해치지 않도록 **감사(Audit)·재분류**할 수 있는가?

### 1.2 연구 목표와 기여

- **목표 1(예측)**: 구간별(p20~p100 등) 데이터를 사용하여 목표값을 예측하고, **Best-K**(상위 K개 특징) 선택 전략으로 성능을 비교한다.
- **목표 2(특징 분석)**: PM/VOC 원천별 및 파생 특징별(예: VOC Lag/Roll, Interaction Sync/Lag 등) 분포를 구간별로 시각화한다.
- **목표 3(검증/감사)**: 상호작용 특징이 실제로 단일 특징의 변형(상수 곱, 완전 상관 등)으로 “가짜 상호작용”이 되지 않았는지 전수 조사하여 재분류한다.
- **목표 4(재현성)**: 폴더 단위 실험 파이프라인을 정리하여 후속 실험(시간 간격 변경 등)을 반복 수행 가능하도록 한다.

### 1.3 보고서 구성

2장에서는 데이터와 타겟 정의를, 3장에서는 특징공학을, 4장에서는 모델과 평가 방법을, 5~6장에서는 실험 설계와 결과를 정리한다. 7장은 본 연구의 핵심 품질관리 절차인 상호작용 특징 감사 및 재분류를 상세히 다룬다. 8장은 해석과 한계를 논의하고, 9장에서 결론과 향후 과제를 제시한다.

---

## 2. 데이터 및 전처리

### 2.1 데이터 구성 및 파일 구조

`YML/ERC/` 폴더에는 원천 데이터 및 전처리 결과, 분석 결과가 다음과 같이 구성되어 있다.

- **원천/기초 데이터**
  - `ERC/PM_timeresampling/*.xlsx`: PM 시계열(예: `Num_0.3um` 포함). 파일 예시: `data1_resampling.xlsx`
  - `ERC/VOC/*.xlsx`: VOC 시계열. 파일 예시: `data1.xlsx`
  - `ERC/Printing_qualitydata.xlsx`: 품질(거칠기) 라벨이 포함된 파일(열: `Roughness(nm)` 등)
- **기존 분석 결과(거칠기 타겟 포함)**
  - `ERC/07_Comparative_Analysis_Roughness/*`: 거칠기 타겟 기반 고정 K 및 분포 분석 결과(엑셀/PNG)
- **최신 실험 폴더(최종 결과 기준)**
  - `ERC/20260115_3,5,10/*`: **lag/window=[3,5,10]**, **p10~p100 포함**, **실제 Roughness(nm) 타겟 병합** 실험(코드/데이터/결과)
- **백업(이전 실험 폴더)**
  - `ERC/20260114_10,60,110/*`, `ERC/20260114_3,5,10/*`: 과거 파이프라인/중간 결과 보존용(본 최종 성능 표의 1차 근거는 아님)

### 2.2 공정 구간(p10~p100) 정의

대부분의 파이프라인은 공정 진행률을 **10% 단위로 절단**하여(예: p10, p20, …, p100) 각 구간의 시계열 조각으로부터 특징을 추출한다. 예를 들어 `20260115_3,5,10`의 Step1 코드에서는 다음과 같이 처리한다.

- 각 샘플의 시계열 길이를 `min_len`으로 맞춘 뒤,
- 구간 `pct`에 대해 `end_idx = max(5, int(min_len * pct/100))` 만큼 앞에서 잘라 사용
- 해당 구간 조각을 long-format으로 변환 후 tsfresh에 입력

이로 인해 “p20 모델”은 공정의 앞 20%에 해당하는 시계열 조각만을 입력으로 사용한다.

### 2.3 결측치 처리 및 주의점(bfill 등)

VOC Lag 특징은 `shift(lag)`로 생성되며 시계열 앞부분에 결측이 발생한다. 최신 실험 코드에서는 주로 `fillna(method='bfill')`로 결측을 채웠다. 이는 길이를 보존하고 모델 입력을 유지하는 장점이 있으나,

- **물리적 의미 관점**: 초반 구간에서 “미래 값으로 과거를 채우는” 형태가 될 수 있음
- **통계적 관점**: 모델에 과도한 평탄화(초기 값 반복)를 유발할 수 있음

따라서 보고서에서는 bfill 사용 이유(길이 보존, 학습 가능성)와 한계를 명시하고, 향후에는 0 채움/구간 삭제/모델이 결측을 다루는 방식(예: 마스킹)을 비교하는 과제를 제시한다.

### 2.4 타겟(Target) 정의 및 정합성 확보(실제 Roughness 병합)

본 프로젝트는 과거 단계에서 “타겟 컬럼 이름은 `Target_Roughness`인데 실제 값은 `Num_0.3um__mean`(PM 평균)”처럼 **타겟 정의가 혼동될 수 있는 상태**가 존재했다. 최종 실험(`ERC/20260115_3,5,10`)에서는 이 문제를 해결하기 위해 다음과 같이 타겟을 **명시적으로 정합화**했다.

- **실제 타겟(최종 학습 타겟)**: `ERC/Printing_qualitydata.xlsx`의 `Roughness(nm)`  
  - 병합 스크립트(최종 실험 전용): `ERC/20260115_3,5,10/code/step3_1_merge_roughness_target.py`  
  - 결과: `ERC/20260115_3,5,10/data/dataset_step3_integrated_roughness.pkl` 내 `Target_Roughness`
- **비교용 Proxy(보존 목적)**: Step3가 생성하던 PM 평균값(기존 `Target_Roughness`에 들어있던 값)을 **`Target_PM_Mean_Proxy`**로 별도 저장  
  - 목적: “PM 평균 기반 예측”과 “실제 거칠기 예측”을 추후 비교/디버깅할 수 있도록 값 보존(단, 학습 feature로는 사용하지 않음)

즉, 최종 결과(성능/중요 특징/분포/빈도/공통 특징)는 **실제 Roughness(nm)** 타겟을 기준으로 산출되었다.

---

## 3. 특징 추출 및 특징공학

### 3.1 `tsfresh` 자동 특징추출 개요

`tsfresh`는 시계열로부터 평균, 분산, 자기상관, 스펙트럼 계수(FFT), 누적분포 기반 지표 등 다양한 특징을 자동으로 추출한다. 본 프로젝트에서는 주로 `EfficientFCParameters()` 설정을 사용하여 비교적 다양한 특징을 생성한다.

기본적인 입력 형태는 long-format으로, 각 시계열 조각에 대해

- `id`: 샘플-구간 식별자(예: `data1_20`)
- `time`: 시간 인덱스(0..N-1)
- `kind`: 변수 이름(예: `VOC`, `VOC_Lag_10s`, `Num_0.3um`)
- `value`: 해당 시점의 값

을 구성하여 `extract_features()`에 전달한다.

### 3.2 VOC 파생 특징: Lag / Rolling / Rolling+Lag

VOC 파생 특징은 다음 3종류로 정의된다.

- **Lag Only**: `VOC_Lag_{lag}s = VOC.shift(lag)`
- **Rolling Only**: `VOC_Roll_{win}s = rolling_mean(VOC, window=win)`
- **Rolling + Lag**: `VOC_Roll_{win}s_Lag_{lag}s = rolling_mean( VOC.shift(lag), window=win )`

기존 분석(`ERC/code/Dataprocess/02_generate_enhanced_features.py`)에서는 **(lag=240,300초)** 및 **(win=180,240초)**를 사용하여 “수 분 단위 지연/누적”을 반영했고, 최신 실험에서는 lag/window를 (10,60,110) 또는 (3,5,10)으로 줄여 **더 미세한 시간 간격**을 탐색하였다.

### 3.3 PM–VOC 상호작용 특징: Product 기반 통합 특징

최신 실험(`20260114_*`)의 상호작용은 Product만 생성한다.

- **정의**: `Prod_PM_x_{VOC_feature} = PM_mean * VOC_feature`
- 여기서 PM_mean은 tsfresh 특징 중 `Num_0.3um__mean`을 대표값으로 사용한다.

이 방식은 구현이 단순하고 상호작용 규모가 큰 장점이 있으나, “PM_mean이 거의 상수처럼 행동하거나, VOC_feature가 특정 상수/항등 특징이면” 상호작용이 단독 특징과 사실상 동일해질 위험이 있어, 7장에서 감사(Audit) 절차로 이를 보정한다.

### 3.4 특징 타입(Type) 분류 체계 및 라벨링

최신 실험의 `Top_Features_Analysis.xlsx`(또는 `_SVR_Perm.xlsx`)에는 각 특징의 `Type`이 저장된다.

- **PM_Raw**: `Num_0.3um` 관련 특징(상호작용 제외)
- **VOC_Raw**: 순수 `VOC` 특징(roll/lag 없음)
- **VOC_Enhanced (Rolling/Lag/Rolling+Lag)**: `VOC_Lag_*`, `VOC_Roll_*`, `VOC_Roll_*_Lag_*` 계열
- **Interaction (Sync/Lag/Roll/Roll+Lag)**: `Prod_`로 시작하는 상호작용 특징(피처명에 Lag/Roll 포함 여부로 세분)

또한 시각화 스크립트에서는 엑셀의 `Type`을 기반으로 그래프 라벨(예: `VOC Roll`, `Int Sync`)을 결정하도록 수정되어, 문자열 매칭에 의존해 발생하던 `Int enh` 오분류를 제거하였다.

---

## 4. 모델링 및 평가 방법

### 4.1 ExtraTrees 회귀(특징 중요도: MDI)

ExtraTreesRegressor는 다수의 랜덤화된 결정트리를 앙상블로 구성하며, 빠르게 특징 중요도를 산출할 수 있다.

- **중요도**: MDI(Mean Decrease in Impurity) 기반 `feature_importances_`
- **장점**: 빠른 학습/해석 가능, 비선형 관계 포착
- **주의**: 고상관 특징이 많을 때 중요도 편향 가능

### 4.2 SVR(RBF) + Target Scaling

RBF 커널 SVR은 입력과 타겟의 스케일에 민감하다. 최신 실험에서는

- 입력 X: `StandardScaler()`
- 타겟 y: `StandardScaler()`를 **CV 내부(train fold)**에서 fit 후 inverse-transform으로 예측을 원 스케일로 복원

을 적용하여 성능을 개선하였다.

### 4.3 Permutation Importance 기반 SVR 특징 랭킹

RBF-SVR은 `coef_` 기반 중요도를 제공하지 않으므로, permutation importance를 사용한다.

- **전략(계산량 절감)**: ExtraTrees(MDI)로 상위 100개 후보를 먼저 뽑은 뒤, 그 100개에 대해 SVR permutation importance 계산
- **반복 수**: `n_repeats=5` (폴더 실험 설정에 따름)

### 4.4 평가 지표: RE(Relative Error)

본 프로젝트에는 RE 정의가 두 형태로 존재한다.

- **(형태 1: 비율)**  
  \n\( RE = \\frac{|y-\\hat{y}|}{|y|} \\)\n  
  예: `ERC/code/Model/05_run_fixed_k_roughness.py`는 평균 RE를 0.10처럼 “비율”로 저장
- **(형태 2: %)**  
  \n\( RE(\\%) = \\frac{|y-\\hat{y}|}{|y|} \\times 100 \\)\n  
  예: `ERC/20260115_3,5,10/code/step4_model_analysis.py` 계열은 %로 저장

본 보고서의 표/그림 해석에서는 **각 결과 파일이 어떤 단위를 사용하는지**를 함께 명시한다.

### 4.5 Best-K 선택(구간별 K 최적화)

각 구간에서 후보 K(`K_CANDIDATES`)를 순회하며 RE를 최소화하는 K를 선택한다.  
이때 모델별로 특징 랭킹 방식이 다르다.

- ExtraTrees: MDI 랭킹 상위 K개
- SVR: (후보 100개 내) permutation 랭킹 상위 K개

---

## 5. 실험 설계

### 5.1 기존 파이프라인(Enhanced/TSFRESH/거칠기 타겟 결합)

기존 파이프라인은 대략 다음 순서로 구성된다.

- **VOC 증강(수 분 단위 lag/rolling)**: `ERC/code/Dataprocess/02_generate_enhanced_features.py`
  - lag=[240,300], window=[180,240]
  - 결과: `ERC/Enhanced_Data/*.xlsx`
- **TSFRESH 특징 추출**: `ERC/code/Dataprocess/03_step1_extract_pm_voc.py` (및 관련 스크립트)
  - 결과: 구간별 특징과 메타(타겟 등) 결합
- **거칠기 타겟 병합**: `ERC/code/Dataprocess/04_merge_roughness_target.py`
  - 결과: `ERC/Enhanced_Features_Roughness/*.pkl`
- **고정 K 평가/시각화**: `ERC/code/Model/05_run_fixed_k_roughness.py`
  - 결과: `ERC/Comparative_Analysis_Roughness/*` (엑셀/그림)

### 5.2 최신 실험 파이프라인(20260115_3,5,10: 실제 Roughness 타겟)

최종 실험은 폴더 단위로 독립 실행 가능한 Step1~Step5 파이프라인을 갖는다.

- `ERC/20260115_3,5,10/code/*`
  - lag/window=[3,5,10]
  - 구간: **p10~p100 포함**
  - 타겟: `Printing_qualitydata.xlsx`의 **`Roughness(nm)` 병합**

공통적으로 샘플 `10,19,20`은 제외하며(총 24개 샘플), 구간 10개(p10~p100)를 사용하여 총 240개(샘플×구간) 인스턴스로 학습/평가한다.

### 5.3 비교 케이스: PM_Only / VOC_Only / Integrated

각 구간에서 다음 3개 데이터 구성에 대해 성능을 평가한다.

- **PM_Only**: PM(Num_0.3um 계열) 특징만 사용
- **VOC_Only**: VOC 및 VOC 증강 특징만 사용
- **Integrated**: PM + VOC + Interaction(Prod_) 특징을 모두 사용

---

## 6. 결과

### 6.1 Best-K 성능 비교(요약)

최신 실험의 핵심 결과는 `Top_Features_Analysis.xlsx`(또는 `_SVR_Perm.xlsx`)의 `Fixed_K_Performance` 시트에 저장된다. 아래 표는 구간별 Best-K를 적용했을 때의 평균 RE(%)를 모델/케이스별로 평균낸 값이다.

#### 6.1.1 최신 실험 요약(평균 RE, %)

**(실험: 20260115_3,5,10; 실제 Roughness(nm) 타겟; p10 포함)** — 파일: `ERC/20260115_3,5,10/results/Top_Features_Analysis.xlsx`

| Model | Case | 평균 RE(%) |
|---|---|---:|
| ExtraTrees | Integrated | 7.2022 |
| ExtraTrees | PM_Only | 7.0606 |
| ExtraTrees | VOC_Only | 7.6135 |
| SVR | Integrated | 7.4385 |
| SVR | PM_Only | 8.4257 |
| SVR | VOC_Only | 7.5568 |

**해석**

- **Integrated가 일관되게 가장 낮은 RE**를 보이며, PM과 VOC를 함께 사용할 때 예측 성능이 향상됨을 시사한다.
- **ExtraTrees가 SVR보다 평균 RE가 낮음**. 이는 트리 기반 앙상블이 본 데이터의 비선형성과 특징 스케일/상관 구조에 더 견고했을 가능성을 의미한다.
- **VOC_Only는 가장 높은 RE**를 보이며, 단독 VOC 특징만으로는 타겟 설명력이 제한적이거나, 특징 설계·타겟 정의·전처리(예: 결측 처리) 개선이 필요함을 시사한다.

#### 6.1.2 Integrated 케이스의 구간별 Best-K(표)

**(실험: 20260115_3,5,10, Integrated)** — 구간별 Best K 및 RE(%)

| Segment(%) | Model | Best K | RE(%) |
|---:|---|---:|---:|
| 10 | ExtraTrees | 100 | 7.0794 |
| 20 | ExtraTrees | 90 | 6.0420 |
| 30 | ExtraTrees | 80 | 6.6172 |
| 40 | ExtraTrees | 50 | 6.5850 |
| 50 | ExtraTrees | 70 | 6.6435 |
| 60 | ExtraTrees | 80 | 7.5608 |
| 70 | ExtraTrees | 70 | 6.9531 |
| 80 | ExtraTrees | 70 | 8.2237 |
| 90 | ExtraTrees | 50 | 7.6995 |
| 100 | ExtraTrees | 70 | 8.6174 |
| 10 | SVR | 10 | 8.6407 |
| 20 | SVR | 50 | 6.0491 |
| 30 | SVR | 50 | 7.7728 |
| 40 | SVR | 40 | 6.1808 |
| 50 | SVR | 70 | 7.1807 |
| 60 | SVR | 50 | 6.2114 |
| 70 | SVR | 60 | 7.0585 |
| 80 | SVR | 30 | 8.1038 |
| 90 | SVR | 60 | 8.4353 |
| 100 | SVR | 40 | 8.7522 |

### 6.2 특징 타입 분포(Best-K 기준)

최신 실험에서는 구간별 Best-K를 기준으로 Top feature 집합을 구성한 뒤, `Type`을 기반으로 상세 라벨을 만들고 누적 막대그래프로 분포를 시각화한다.

- 결과 이미지(최종 실험)
  - `ERC/20260115_3,5,10/results/Feature_Distribution_BestK_ExtraTrees.png`
  - `ERC/20260115_3,5,10/results/Feature_Distribution_BestK_SVR.png`

본 분포 그래프는 다음 질문에 답한다.

- 공정 초기/중기/후기 구간에서 **PM Raw vs VOC Enhanced vs Interaction** 중 무엇이 더 많이 선택되는가?
- Interaction 중에서도 **Sync(동시) vs Lag/Roll(시간 지연/누적)**의 비중이 어떻게 변하는가?

### 6.3 공통 Top 특징(모델 교집합) 및 유사도(Jaccard)

`ERC/20260115_3,5,10/code/extract_common_features.py`는 Integrated 케이스에서 각 구간의 Best-K를 기준으로 SVR과 ExtraTrees의 Top feature 교집합을 구하고, 다음을 저장한다.

- `Common_Top_Features.xlsx`
  - `Common_Features_Detail`: 교집합 특징, 모델별 순위, 평균 순위
  - `Summary_Stats`: 교집합 개수, Jaccard similarity 등

이 분석은 “모델이 달라도 일관되게 중요한 특징”을 찾는 데 목적이 있다.

### 6.4 구간 간 지속성(빈도) 분석

빈도 분석은 “여러 구간에서 반복적으로 중요하게 선택되는 특징”을 찾아 순위를 매긴다.

- SVR 지속성: `SVR_Features_Frequency.xlsx`
- ExtraTrees 지속성: `ExtraTrees_Features_Frequency.xlsx`
- 공통 특징 지속성: `Common_Features_Frequency.xlsx`

이 결과는 후속 연구에서 “핵심 특징 후보(robust features)”를 선별하는 기준으로 사용할 수 있다.

### 6.5 (참고) 기존 거칠기 타겟 실험 결과

`ERC/Comparative_Analysis_Roughness/Fixed_K_Roughness_Result.xlsx`는 거칠기 타겟 기반 고정 K 실험 결과를 제공한다.  
단, 여기서의 Mean_RE는 **비율 단위(예: 0.10)**로 저장되어 있으며, 최신 실험의 RE(%)와 단위가 다름에 유의해야 한다.

요약(평균 Mean_RE, 비율):

| Model | Case | 평균 Mean_RE |
|---|---|---:|
| SVR | Integrated | 0.0958 |
| SVR | PM_Only | 0.1020 |
| SVR | VOC_Only | 0.0958 |
| ExtraTrees | Integrated | 0.1013 |
| ExtraTrees | PM_Only | 0.1051 |
| ExtraTrees | VOC_Only | 0.1053 |

---

## 7. 상호작용 특징 감사(Audit) 및 재분류

### 7.1 “가짜 상호작용” 발생 메커니즘

상호작용 특징이 진정한 시너지/결합 효과를 담기 위해서는 PM과 VOC의 변동이 함께 반영되어야 한다. 그러나 자동 특징 추출(`tsfresh`) 및 파생 특징 생성 과정에서 다음과 같은 특징들이 생성될 수 있다.

- **상수/항등 특징**: 예) `autocorrelation__lag_0`은 시계열이 비어있지 않으면 1에 가까워지는 경우가 있어, 어떤 값과 곱해도 사실상 원래 값이 됨
- **완전상관 특징**: 상호작용 결과가 PM_mean 또는 VOC_feature와 거의 동일한 형태가 되는 경우

이 경우 “Interaction으로 분류되었지만 실제로는 PM 단독/ VOC 단독 특징”이며, 분석 결과(분포/빈도/공통 특징)가 왜곡될 수 있다.

### 7.2 전수 감사 절차 및 기준(상관계수 0.9999)

최신 실험 폴더에는 감사 스크립트가 포함된다.

- `ERC/20260115_3,5,10/code/audit_interaction_features.py` (최종 실험)
- (참고/백업) `ERC/20260114_3,5,10/code/audit_interaction_features.py`, `ERC/20260114_10,60,110/code/audit_interaction_features.py`

감사 절차는 다음과 같다.

- 상호작용 특징 집합: `Prod_PM_x_*`
- 기준 PM 특징: `Num_0.3um__mean`
- 각 상호작용 특징에 대해:
  - corr(Interaction, PM_mean) > 0.9999 이면 **PM_Only로 재분류**
  - corr(Interaction, 해당 VOC 파트) > 0.9999 이면 **VOC로 재분류**

### 7.3 재분류 규칙(A안 포함) 및 결과 반영

재분류가 VOC로 넘어가는 경우, VOC 파생 여부(Lag/Roll)가 명시되지 않은 특징이 존재할 수 있다. 본 연구에서는 사용자 합의에 따라 **A안**을 적용한다.

- **A안(채택)**: feature 이름에 Roll/Lag 표기가 없으면 `VOC_Raw`로 분류
  - Roll+Lag 모두 존재: `VOC_Enhanced (Rolling+Lag)`
  - Roll만 존재: `VOC_Enhanced (Rolling Only)`
  - Lag만 존재: `VOC_Enhanced (Lag Only)`
  - 둘 다 없음: `VOC_Raw`

이 규칙은 `ERC/20260115_3,5,10/code/update_top_features_analysis_types.py`로 구현되어 `Top_Features_Analysis.xlsx`에 감사 결과를 명시적으로 기록하였다(백업 포함).

또한 최종 실험(`ERC/20260115_3,5,10/results/Top_Features_Analysis.xlsx`)의 `Audit_Results` 기준으로, 상호작용 특징 중 **346개**가 재분류 대상으로 식별되었고(**PM_Raw 306개**, **VOC_Enhanced 40개**), 이는 “상호작용 분포/빈도 분석”에서 오해를 줄이는 핵심 근거가 된다.

### 7.4 시각화 라벨(`Int enh`) 문제와 해결

과거 시각화 로직에서 `Type`을 신뢰하지 않고 문자열 기반으로 분류하는 과정에서 `Interaction (Sync)`가 “Lag/Roll로 판별되지 않아” 임의 카테고리(`Int enh`)로 떨어지는 문제가 발생했다.  
최신 실험에서는 **엑셀의 `Type`을 우선** 사용하고, `Interaction (Sync)`는 항상 `Int Sync`로 매핑하여 문제를 제거했다.

---

## 8. 논의

### 8.1 지연/누적 효과의 해석: 왜 과거 VOC가 도움이 될 수 있는가

VOC의 변화가 즉시 PM 변화 또는 품질 변화로 반영되지 않고, 물리·화학적 과정(응축, 입자화, 확산, 공정 열역학)에 의해 **지연**될 수 있다면, “현재 시점”의 VOC보다 “과거의 VOC” 또는 “최근 일정 구간의 평균 VOC”가 더 유의미한 예측 신호가 될 수 있다.  
본 프로젝트의 VOC Lag/Roll 특징은 이러한 가설을 정량적으로 탐색하기 위한 도구이며, 상호작용(Prod) 특징은 PM과 VOC의 결합 효과를 모델이 학습할 수 있도록 한다.

다만 본 보고서는 데이터만으로 인과를 단정하지 않으며, “지연 특징이 성능 개선에 기여할 수 있다”는 경험적 관찰과 그 가능성에 대한 물리적 해석을 제시하는 수준으로 정리한다.

### 8.2 모델별 성능 차이(ExtraTrees vs SVR) 해석

최신 실험에서 ExtraTrees가 SVR보다 평균 RE가 낮은 경향을 보였다. 가능한 이유는 다음과 같다.

- 트리 앙상블은 상호작용과 비선형을 자연스럽게 모델링하며 스케일링 민감도가 낮다.
- SVR은 커널 파라미터(C, epsilon, gamma) 및 스케일링에 민감하며, 데이터 크기/노이즈/특징 수가 많을 때 적절한 튜닝이 필요하다.
- SVR permutation importance는 계산량이 커서 후보를 100개로 제한했는데, 이 과정이 SVR에게 최적 특징 집합을 제한했을 가능성도 존재한다.

### 8.3 연구 타당성/재현성 관점 점검

본 프로젝트는 분석과정에서 다음의 “연구 품질 체크포인트”를 드러낸다.

- **타겟 정의 일관성**: “거칠기 타겟” 파이프라인과 “PM 평균 타겟” 파이프라인이 혼재 → 보고서에서 명확히 구분 필요
- **RE 단위 혼재**: 비율 vs % 혼재 → 결과 비교 시 단위 통일 또는 명시 필요
- **상호작용 특징 오분류**: 가짜 상호작용 존재 → 감사(Audit) 및 재분류로 해결
- **결측 처리(bfill)의 의미**: 초반 데이터 처리에 대한 민감도 분석 필요(향후 과제)

---

## 9. 결론 및 향후 연구

### 9.1 결론

- 공정 구간별(p20~p100) 특징 기반 예측에서 **Integrated(PM+VOC+Interaction)**가 일관되게 가장 낮은 RE를 보였다.
- **ExtraTrees(MDI)**가 본 데이터에서 **SVR(RBF)** 대비 더 낮은 평균 RE를 기록했다.
- 자동 생성 상호작용 특징에는 “실제로는 단독 특징”인 항목이 섞일 수 있으며, **감사(Audit) 및 재분류**는 특징 분포/빈도 분석의 타당성을 위해 필수적이다.

### 9.2 향후 연구

- **(타겟 정합성)** 최신 실험 파이프라인에도 `Printing_qualitydata.xlsx` 기반 실제 `Roughness(nm)` 타겟을 병합하여 “진짜 거칠기 예측” 실험을 재수행
- **(결측/경계 처리)** lag 특징의 초기 구간 결측 처리(bfill/0/drop/마스킹) 비교
- **(상호작용 확장)** Product 외에 Ratio, Sum, Diff 등 다양한 상호작용을 포함하되, 감사 절차를 함께 적용
- **(모델 튜닝/검증)** SVR의 하이퍼파라미터 탐색 및 데이터 분할 전략(LOOCV vs KFold) 비교
- **(해석 가능성 강화)** permutation importance 외에 SHAP 등 해석 도구 도입

---

## 참고문헌(초안)

- Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). *Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh)*.
- Breiman, L. (2001). *Random Forests*. (MDI 중요도 관련 배경)
- Fisher, A., Rudin, C., & Dominici, F. (2019). *All Models are Wrong, but Many are Useful: Learning a Variable’s Importance by Studying an Entire Class of Prediction Models*. (Permutation 기반 해석 논의)
- Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. (SVR 배경)

---

## 부록 A. 폴더별 실행 가이드(재현성)

### A.1 최신 실험(20260115_3,5,10; 실제 Roughness 타겟) 재현

- 실행 순서(권장)
  - `python ERC/20260115_3,5,10/code/step1_2_extract_features.py`
  - `python ERC/20260115_3,5,10/code/step3_create_datasets.py`
  - `python ERC/20260115_3,5,10/code/step3_1_merge_roughness_target.py`  *(실제 Roughness 타겟 병합)*
  - `python ERC/20260115_3,5,10/code/step4_model_analysis.py`
  - `python ERC/20260115_3,5,10/code/audit_interaction_features.py`
  - `python ERC/20260115_3,5,10/code/update_top_features_analysis_types.py` (A안 정규화/백업)
  - `python ERC/20260115_3,5,10/code/step5_visualization.py`
  - (선택) 공통/빈도 분석
    - `python ERC/20260115_3,5,10/code/extract_common_features.py`
    - `python ERC/20260115_3,5,10/code/analyze_feature_frequency.py`
    - `python ERC/20260115_3,5,10/code/analyze_svr_frequency.py`
    - `python ERC/20260115_3,5,10/code/analyze_et_frequency.py`

### A.2 최신 실험(20260114_10,60,110) 재현

- (SVR permutation 버전)
  - `python ERC/20260114_10,60,110/code/step1_2_extract_features.py`
  - `python ERC/20260114_10,60,110/code/step3_create_datasets.py`
  - `python ERC/20260114_10,60,110/code/step4_svr_perm.py`
  - `python ERC/20260114_10,60,110/code/audit_interaction_features.py`
  - `python ERC/20260114_10,60,110/code/step5_visualization.py`

---

## 부록 B. 주요 결과물(파일 목록)

- 최신 실험(최종, 20260115_3,5,10; 실제 Roughness 타겟)
  - `ERC/20260115_3,5,10/results/Top_Features_Analysis.xlsx`
  - `ERC/20260115_3,5,10/results/Fixed_K_Comparison_SVR.png`
  - `ERC/20260115_3,5,10/results/Fixed_K_Comparison_ExtraTrees.png`
  - `ERC/20260115_3,5,10/results/Feature_Distribution_BestK_SVR.png`
  - `ERC/20260115_3,5,10/results/Feature_Distribution_BestK_ExtraTrees.png`
  - `ERC/20260115_3,5,10/results/Common_Top_Features.xlsx`
  - `ERC/20260115_3,5,10/results/Common_Features_Frequency.xlsx`
  - `ERC/20260115_3,5,10/results/SVR_Features_Frequency.xlsx`
  - `ERC/20260115_3,5,10/results/ExtraTrees_Features_Frequency.xlsx`
- 최신 실험(10,60,110)
  - `ERC/20260114_10,60,110/results/Top_Features_Analysis_SVR_Perm.xlsx`
  - `ERC/20260114_10,60,110/results/Fixed_K_Comparison_SVR.png`
  - `ERC/20260114_10,60,110/results/Fixed_K_Comparison_ExtraTrees.png`
  - `ERC/20260114_10,60,110/results/Feature_Distribution_BestK_SVR.png`
  - `ERC/20260114_10,60,110/results/Feature_Distribution_BestK_ExtraTrees.png`
- 기존 거칠기 분석
  - `ERC/07_Comparative_Analysis_Roughness/Fixed_K_Roughness_Result.xlsx`
  - `ERC/07_Comparative_Analysis_Roughness/Fixed_K_Comparison_*.png`
  - `ERC/07_Comparative_Analysis_Roughness/Feature_Distribution_*.png`


