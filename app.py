import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "hr_attrition_v1.pkl"
META_PATH = BASE / "hr_attrition_v1_meta.json"

st.set_page_config(page_title="퇴사 위험도 예측", page_icon="👥", layout="wide")


def _load_raw():
    try:
        return joblib.load(MODEL_PATH)
    except AttributeError:
        # sklearn 버전 차이로 저장 시점 클래스가 없을 때의 임시 호환 처리
        import sklearn.compose._column_transformer as ct

        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList
        return joblib.load(MODEL_PATH)


@st.cache_resource
def load_model():
    m = _load_raw()
    # 1.6.x로 저장된 모델을 더 새로운 sklearn에서 쓸 때 SimpleImputer 속성 호환 처리
    try:
        for _, t, _ in m.named_steps["pre"].transformers_:
            steps = getattr(t, "named_steps", {})
            for step in steps.values():
                if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
                    step._fill_dtype = step._fit_dtype
    except Exception:
        pass
    return m


@st.cache_data
def load_meta():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


REF_PATH = BASE / "HR-Employee-Attrition.csv"  # 학습 데이터 (직급별 평균 소득 계산용)
DERIVED = ["PromotionStall", "JobHopRate", "IncomeVsLevel", "BurnoutRisk"]
FE_BASE = ["YearsSinceLastPromotion", "YearsAtCompany", "Companies", "Age",
           "Income_M", "JobLevel", "OverTime", "WorkLifeBalance"]


@st.cache_data
def load_level_means():
    """학습 데이터 기준 JobLevel별 평균 Income_M. 파일이 없으면 None."""
    if REF_PATH.exists():
        ref = pd.read_csv(REF_PATH)
        if {"JobLevel", "Income_M"} <= set(ref.columns):
            return ref.groupby("JobLevel")["Income_M"].mean().to_dict()
    return None


def add_features(data, level_means=None):
    """학습 때 사용한 도메인 지식 기반 파생 피처 생성 (원본은 변경하지 않음)"""
    d = data.copy()
    d["PromotionStall"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["JobHopRate"] = d["Companies"] / (d["Age"] - 18).clip(lower=1)
    if level_means is not None:
        lvl = d["JobLevel"].map(level_means)
    else:
        lvl = d.groupby("JobLevel")["Income_M"].transform("mean")
    d["IncomeVsLevel"] = d["Income_M"] / lvl
    d["BurnoutRisk"] = (d["OverTime"] == "Yes").astype(int) * (5 - d["WorkLifeBalance"])
    return d


model = load_model()
meta = load_meta()
LEVEL_MEANS = load_level_means()
COLS = meta["input_columns"]
THRESH = meta["threshold"]

# 학습 데이터 기준 범위/카테고리 (모델 내부 스케일러에서 추출)
NUM_INT = {
    "Income_M": (1009, 19999, 4941),
    "p_SalaryHike": (11, 25, 14),
    "Age": (18, 60, 36),
    "Dist": (1, 29, 7),
    "Companies": (0, 9, 2),
    "YearsAtCompany": (0, 37, 5),
    "YearsSinceLastPromotion": (0, 15, 1),
    "BurnoutRisk": (0, 4, 0),
}
NUM_FLOAT = {
    "PromotionStall": (0.0, 1.0, 0.14),
    "JobHopRate": (0.0, 1.5, 0.11),
    "IncomeVsLevel": (0.3, 1.9, 0.99),
}
ORD = {
    "Education": (1, 5, 3),
    "JobLevel": (1, 5, 2),
    "JobSatisfaction": (1, 4, 3),
    "Env_Satisfaction": (1, 4, 3),
    "RelationshipSatisfaction": (1, 4, 3),
    "Performance": (1, 4, 3),
    "StockOption": (0, 3, 1),
    "WorkLifeBalance": (1, 4, 3),
}
CAT = {
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "Gender": ["Female", "Male"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}


def predict(df: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(df[COLS])[:, 1], index=df.index)


def risk_label(p: float) -> str:
    return "🔴 고위험 (면담 대상)" if p >= THRESH else "🟢 일반"


st.title("👥 직원 퇴사 위험도 예측")
st.caption(
    f"모델 {meta['model_version']} · {meta['algorithm']} · "
    f"고위험 기준 확률 ≥ {THRESH:.0%}"
)

tab_single, tab_batch, tab_info = st.tabs(["개별 예측", "엑셀 일괄 예측", "모델 정보"])

# ---------------------------------------------------------------- 개별 예측
with tab_single:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("기본 정보")
            row = {}
            for k, v in CAT.items():
                row[k] = st.selectbox(k, v)
            row["Age"] = st.slider("Age", *NUM_INT["Age"][:2], NUM_INT["Age"][2])
            row["Dist"] = st.slider("Dist (통근 거리)", *NUM_INT["Dist"][:2], NUM_INT["Dist"][2])
            row["Education"] = st.slider("Education", *ORD["Education"][:2], ORD["Education"][2])
            row["JobLevel"] = st.slider("JobLevel", *ORD["JobLevel"][:2], ORD["JobLevel"][2])

        with c2:
            st.subheader("경력 · 보상")
            row["Income_M"] = st.number_input("Income_M (월 소득)", 1009, 19999, NUM_INT["Income_M"][2])
            row["p_SalaryHike"] = st.slider("p_SalaryHike (%)", *NUM_INT["p_SalaryHike"][:2], NUM_INT["p_SalaryHike"][2])
            row["Companies"] = st.slider("Companies (이전 근무 회사 수)", *NUM_INT["Companies"][:2], NUM_INT["Companies"][2])
            row["YearsAtCompany"] = st.slider("YearsAtCompany", *NUM_INT["YearsAtCompany"][:2], NUM_INT["YearsAtCompany"][2])
            row["YearsSinceLastPromotion"] = st.slider(
                "YearsSinceLastPromotion", *NUM_INT["YearsSinceLastPromotion"][:2], NUM_INT["YearsSinceLastPromotion"][2]
            )
            row["StockOption"] = st.slider("StockOption", *ORD["StockOption"][:2], ORD["StockOption"][2])

        with c3:
            st.subheader("만족도 · 업무")
            for k in ["JobSatisfaction", "Env_Satisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "Performance"]:
                row[k] = st.slider(k, *ORD[k][:2], ORD[k][2])

        manual_ivl = None
        if LEVEL_MEANS is None:
            st.warning(
                "HR-Employee-Attrition.csv를 찾지 못해 직급별 평균 소득을 계산할 수 없습니다. "
                "IncomeVsLevel(직급 평균 대비 소득 비율)을 직접 입력해 주세요."
            )
            manual_ivl = st.number_input("IncomeVsLevel", 0.3, 1.9, 0.99, step=0.01)
        st.caption("PromotionStall, JobHopRate, IncomeVsLevel, BurnoutRisk는 위 입력값으로 자동 계산됩니다.")

        submitted = st.form_submit_button("예측하기", type="primary")

    if submitted:
        one = pd.DataFrame([row])
        one = add_features(one, LEVEL_MEANS if LEVEL_MEANS is not None else {row["JobLevel"]: row["Income_M"]})
        if manual_ivl is not None:
            one["IncomeVsLevel"] = manual_ivl
        p = float(predict(one)[0])
        m1, m2 = st.columns(2)
        m1.metric("퇴사 위험 확률", f"{p:.1%}")
        m2.metric("판정", risk_label(p))
        st.progress(min(max(p, 0.0), 1.0))

# ---------------------------------------------------------------- 일괄 예측
with tab_batch:
    st.write(
        "엑셀(.xlsx) 또는 CSV 파일을 올리면 전 직원의 위험 확률을 계산합니다. "
        "아래 컬럼이 필요하며, 파생 변수(PromotionStall, JobHopRate, IncomeVsLevel, BurnoutRisk)는 "
        "없으면 자동으로 계산합니다."
    )
    st.code(", ".join(COLS), language="text")

    up = st.file_uploader("파일 업로드", type=["xlsx", "xls", "csv"])
    if up is not None:
        df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        need_fe = [c for c in DERIVED if c not in df.columns]
        if need_fe:
            base_missing = [c for c in FE_BASE if c not in df.columns]
            if not base_missing:
                fe = add_features(df, LEVEL_MEANS)
                for c in need_fe:
                    df[c] = fe[c]
                st.info(f"파생 변수 자동 생성: {', '.join(need_fe)}")
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            st.error(f"누락된 컬럼: {', '.join(missing)}")
        else:
            out = df.copy()
            out["attrition_proba"] = predict(df).round(4)
            out["risk_group"] = (out["attrition_proba"] >= THRESH).map({True: "고위험", False: "일반"})

            n_high = int((out["risk_group"] == "고위험").sum())
            a, b, c = st.columns(3)
            a.metric("전체 인원", f"{len(out):,}")
            b.metric("고위험 인원", f"{n_high:,}")
            c.metric("고위험 비율", f"{n_high / len(out):.1%}")

            out = out.sort_values("attrition_proba", ascending=False)
            st.dataframe(out, use_container_width=True, height=400)

            buf = io.BytesIO()
            out.to_excel(buf, index=False)
            st.download_button(
                "결과 엑셀 다운로드",
                buf.getvalue(),
                file_name="attrition_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ---------------------------------------------------------------- 모델 정보
with tab_info:
    st.subheader("성능 (검증 데이터)")
    a, b, c = st.columns(3)
    a.metric("Recall", f"{meta['valid_recall']:.1%}")
    b.metric("Precision", f"{meta['valid_precision']:.1%}")
    c.metric("ROC-AUC", f"{meta['valid_roc_auc']:.3f}")
    st.subheader("임계값 근거")
    st.write(f"**{THRESH}** — {meta['threshold_basis']}")
    st.caption(f"학습일 {meta['trained_at']} · random_state {meta['random_state']}")
    with st.expander("입력 컬럼 전체"):
        st.write(COLS)
