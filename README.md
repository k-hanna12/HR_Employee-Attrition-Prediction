# 👥 직원 퇴사 위험도 예측

HR 데이터로 **퇴사 가능성이 높은 직원을 미리 찾아내는** 모델입니다.
인사팀이 고위험 직원과 먼저 상담해 핵심 인재 이탈과 채용 비용을 줄이는 것이 목표입니다.

<br>

## 1. 모델 개발 배경 및 목표

**배경**
퇴사 의사는 보통 면담 시점에야 알게 되어, 그때는 되돌리기 어렵습니다.
하지만 야근 여부, 급여, 만족도 같은 HR 데이터에는 이미 퇴사 징후가 담겨 있습니다.

**목표**
- 직원별 퇴사 확률을 예측해 고위험군을 선별
- 인사팀의 선제적 리텐션 상담 지원

**평가 지표**
퇴사자 비율이 약 16%로 불균형하므로, 정확도 대신 **Recall**과 **ROC-AUC**를 기준으로 삼았습니다.
퇴사자를 놓치는 비용이 더 크기 때문입니다.

<br>

## 2. 피쳐 엔지니어링

- **데이터:** IBM HR 데이터셋 (1,470명, 35개 변수)
- **제거:** 값이 하나뿐인 컬럼(`EmployeeCount`, `StandardHours`, `Over18`), 식별자(`EmployeeNumber`)
- **인코딩:** 범주형은 One-Hot, `OverTime`·`Gender`는 0/1
- **스케일링:** 수치형 변수 StandardScaler
- **불균형 처리:** <!-- TODO --> class_weight 또는 SMOTE

<details>
<summary>파생 변수 보기</summary>

<!-- TODO: 실제 만든 변수로 수정 -->
| 변수 | 의미 |
|---|---|
| `IncomePerLevel` | 직급 대비 급여 |
| `YearsPerCompany` | 이직 빈도 |
| `SatisfactionScore` | 종합 만족도 |

</details>

<br>

## 3. 파이프라인

```
데이터 정제 → 피쳐 엔지니어링 → 학습/검증 분리 → 모델 학습·튜닝 → 모델 저장 → 웹 앱 서빙
```

- 학습된 모델은 `hr_attrition_v1.pkl`, 피처 목록·성능 정보는 `hr_attrition_v1_meta.json`에 저장
- `app.py`가 모델을 불러와 예측 결과를 보여줍니다

**최종 성능** <!-- TODO -->
| ROC-AUC | Recall |
|:---:|:---:|
| 0.818 | 29.6% |

<br>

## 4. 페이지 화면

<img width="2742" height="1420" alt="hr_pc1" src="https://github.com/user-attachments/assets/8f55ee7a-c1be-4db7-b5fc-b3e3d9c6b913" />
<img width="2712" height="1386" alt="hr_pc2" src="https://github.com/user-attachments/assets/4c0e9789-9997-4e2a-a694-f44301de4666" />

<br>

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

<br>

> ⚠️ 가상 데이터로 학습한 모델이며, 결과는 **상담 우선순위 참고용**입니다.
> 인사 평가나 불이익 판단에는 사용하지 않습니다.

---------------------------------

## 👥 Employee Attrition Risk Prediction

A model that uses HR data to identify employees at high risk of leaving. The goal is to help the HR team reach out to high-risk employees first, reducing the loss of key talent and hiring costs.

### 1. Background & Objectives

**Background**
Intentions to resign usually surface only during exit interviews, when it is often too late to change the outcome. However, HR data such as overtime, salary, and satisfaction levels already contain early signs of attrition.

**Objectives**
- Predict each employee's probability of leaving and identify the high-risk group
- Support the HR team with proactive retention conversations

**Evaluation Metrics**
Since only about 16% of employees left, the dataset is imbalanced. We therefore use Recall and ROC-AUC instead of accuracy, because missing an employee who will leave is more costly.

### 2. Feature Engineering

- **Data:** IBM HR Analytics dataset (1,470 employees, 35 variables)
- **Removed:** Constant columns (EmployeeCount, StandardHours, Over18) and the identifier (EmployeeNumber)
- **Encoding:** One-Hot encoding for categorical variables; OverTime and Gender encoded as 0/1
- **Scaling:** StandardScaler for numerical variables
- **Class imbalance:** class_weight or SMOTE

<details>
<summary>View derived features</summary>

</details>

### 3. Pipeline

Data cleaning → Feature engineering → Train/validation split → Model training & tuning → Model saving → Web app serving

- The trained model is saved as `hr_attrition_v1.pkl`, and the feature list and performance metrics are saved in `hr_attrition_v1_meta.json`.
- `app.py` loads the model and displays the prediction results.

**Final Performance**

| ROC-AUC | Recall |
|---|---|
| 0.818 | 29.6% |

### 4. Screenshots

<img width="2742" height="1420" alt="hr_pc1" src="https://github.com/user-attachments/assets/8f55ee7a-c1be-4db7-b5fc-b3e3d9c6b913" />
<img width="2712" height="1386" alt="hr_pc2" src="https://github.com/user-attachments/assets/4c0e9789-9997-4e2a-a694-f44301de4666" />

### How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ This model was trained on fictional data. Its results are intended only as a reference for prioritizing retention conversations and must not be used for performance evaluations or any decisions that disadvantage employees.
