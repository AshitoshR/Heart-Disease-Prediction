import streamlit as st
import numpy as np
import pickle
import joblib
from pathlib import Path

#*_____Page Config_____*
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="HeaRTh",
    layout="wide"
)

#*_____Load model & scaler_____*
@st.cache_resource
def load_models():
    base = Path(__file__).resolve().parent.parent / "models"
    return (
        joblib.load(base / "logistic_model.pkl"),
        joblib.load(base / "scaler.pkl")
    )

model, scaler = load_models()

#*_____Title & Intro_____*
st.title("Heart Disease Prediction System")

st.markdown("""
### Why Heart Health Matters
Heart disease is one of the leading causes of death worldwide.  
Unhealthy lifestyle choices, poor diet, stress, and lack of exercise
significantly increase the risk.

### ⚠️ Disadvantages of Poor Heart Health
- Increased risk of heart attack and stroke  
- Reduced life expectancy  
- Long-term medication dependency  
- Reduced quality of life  

### 💡 Tips to Maintain a Healthy Heart
- Exercise regularly 🏃  
- Eat balanced, low-fat foods 🥗  
- Avoid smoking 🚭  
- Manage stress 🧘  
- Get regular health checkups 🩺     
""")

# Proct root
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"

#*_____Image Section_____*
st.markdown("Heart Health Awareness")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(IMAGE_DIR / "heart.jpg", caption="Heatly Heart", use_container_width=True)

with col2:
    st.image(IMAGE_DIR / "heart.jpg", caption="Heart Care Awareness", use_container_width=True)

with col3:
    st.image(IMAGE_DIR / "heart.jpg", caption="Prevention is better", use_container_width=True)

st.markdown("---")

#*_____Example Patient Data_____*
st.subheader("Select Example Patient Data")

examples = {
    "Healthy Individual": [45, 0, 3, 120, 220, 0, 0, 170, 0, 0.2, 1, 0, 3],
    "Moderate Risk Patient": [55, 1, 4, 135, 260, 0, 1, 145, 1, 1.2, 2, 1, 6],
    "High Risk Patient": [65, 1, 4, 150, 300, 1, 2, 120, 1, 2.8, 2, 3, 7]
}

example_choice = st.selectbox("Choose an example", list(examples.keys()))
example_data = examples[example_choice]

#*_____Input Selection_____*
st.subheader("Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, example_data[0])
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1], index=example_data[1])
    cp = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4], index=example_data[2]-1)
    bp = st.number_input("Resting BP", 80, 200, example_data[3])
    chol = st.number_input("Cholesterol", 100, 600, example_data[4])

with col2:
    fbs = st.selectbox("FBS > 120 (0=No, 1=Yes)", [0,1], index=example_data[5])
    ekg = st.selectbox("EKG Results (0–2)", [0,1,2], index=example_data[6])
    max_hr = st.number_input("Max Heart Rate", 60, 220, example_data[7])
    ex_angina = st.selectbox("Exercise Angina (0=No,1=Yes)", [0,1], index=example_data[8])

with col3:
    st_dep = st.number_input("ST Depression", 0.0, 6.5, float(example_data[9]))
    slope = st.selectbox("Slope of ST (1–3)", [1,2,3], index=example_data[10]-1)
    vessels = st.selectbox("Number of Vessels (0–3)", [0,1,2,3], index=example_data[11])
    thallium = st.selectbox("Thallium (3,6,7)", [3,6,7], index=[3,6,7].index(example_data[12]))

#*_____Prediction_____*
if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, bp, chol, fbs, ekg, max_hr, 
                            ex_angina, st_dep, slope, vessels, thallium]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Heart Disease\nProbability: {prob:.2%}")
    elif prediction == 0:
        st.success(f"Low Risk of Heart Disease\nProbability: {prob:.2%}")
    else:
        st.error(f"Probability: {prob:.2%}")