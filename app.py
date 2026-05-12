import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SmartClinic AI", page_icon="🏥", layout="centered")

st.title("🏥 SmartClinic AI")
st.subheader("Heart Disease Risk Predictor")
st.markdown("Enter patient data below to predict heart disease risk.")

@st.cache_resource
def load_model():
    return joblib.load("outputs/model.pkl")

model = load_model()

st.markdown("---")
st.markdown("### Patient Data")

col1, col2 = st.columns(2)

with col1:
    age      = st.slider("Age", 20, 80, 50)
    sex      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
    cp       = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    restecg  = st.selectbox("Resting ECG (0-2)", [0,1,2])

with col2:
    thalach  = st.slider("Max Heart Rate", 60, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak  = st.slider("ST Depression", 0.0, 7.0, 1.0)
    slope    = st.selectbox("Slope of ST Segment (0-2)", [0,1,2])
    ca       = st.selectbox("Major Vessels Colored (0-3)", [0,1,2,3])
    thal     = st.selectbox("Thalassemia (0-3)", [0,1,2,3])

st.markdown("---")

if st.button("🔍 Predict Risk", type="primary"):
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal,
        'chol_risk': 1 if chol > 240 else 0,
        'low_max_hr': 1 if thalach < 140 else 0
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.markdown("### Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — Probability: {probability:.1%}")
        st.markdown("This patient shows indicators of heart disease.")
    else:
        st.success(f"✅ LOW RISK — Probability: {probability:.1%}")
        st.markdown("This patient shows low indicators of heart disease.")

    st.warning("⚕️ This tool is for decision support only. Always consult a physician.")

st.markdown("---")
st.caption("SmartClinic AI | Hebrew University of Jerusalem | 2026")