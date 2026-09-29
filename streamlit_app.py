import joblib
import pandas as pd
import streamlit as st
from typing import Literal

# Load the same trained model used by the FastAPI application.
MODEL_PATH = "Mental_Health_Model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

TOP_COUNTRIES = [
    "Other", "India", "USA", "Canada", "Australia",
    "UK", "Germany", "Turkey", "Mexico", "France"
]

PLATFORMS = [
    "Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter",
    "YouTube", "TikTok", "LINE", "KakaoTalk", "VKontakte",
    "WhatsApp", "WeChat"
]

PURPOSES = ["Networking", "Education", "Entertainment", "News"]
ACADEMIC_LEVELS = ["Undergraduate", "Graduate", "High_School"]
STRESS_LEVELS = ["Low", "Medium", "High", "Very High"]
GENDERS = ["Male", "Female"]


st.set_page_config(
    page_title="Student Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef4ff 100%);
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #1f4e79 0%, #4f8cc9 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(31, 78, 121, 0.18);
    }
    .hero h1 {
        margin-bottom: 0.4rem;
        font-size: 2.35rem;
    }
    .hero p {
        margin: 0;
        font-size: 1.05rem;
        opacity: 0.92;
    }
    .card {
        background: white;
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        border: 1px solid #e7ebf2;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .result {
        text-align: center;
        padding: 1.7rem;
        border-radius: 20px;
        background: white;
        border: 1px solid #dce5f0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.07);
    }
    .score {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .muted {
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🧠 Student Mental Health Predictor</h1>
    <p>Enter student lifestyle, academic, social-media and stress information to generate a model-based mental health score.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        "The trained model file `Mental_Health_Model.pkl` was not found. "
        "Place it in the same folder as this Streamlit app and restart the app."
    )
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(
        "This interface uses the trained machine-learning model from your "
        "FastAPI project. It sends the same feature names and values used by "
        "the `/predict` endpoint."
    )
    st.divider()
    st.caption("Model file: Mental_Health_Model.pkl")
    st.caption("Frontend: Streamlit")


with st.form("prediction_form"):
    st.subheader("👤 Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", GENDERS)
        country = st.text_input("Country", value="India")

    with col2:
        academic_level = st.selectbox("Academic Level", ACADEMIC_LEVELS)
        most_used_platform = st.selectbox("Most Used Platform", PLATFORMS)
        purpose_of_use = st.selectbox("Purpose of Use", PURPOSES)

    with col3:
        stress_level = st.selectbox("Stress Level", STRESS_LEVELS)
        daily_unlocks = st.number_input(
            "Daily Unlocks", min_value=0, value=50, step=1
        )
        sleep_hours = st.number_input(
            "Sleep Hours / Night", min_value=0.0, max_value=24.0,
            value=7.0, step=0.5
        )

    st.subheader("📊 Daily Habits")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_daily_usage_hours = st.number_input(
            "Average Daily Social Media Usage (hours)",
            min_value=6.0,
            max_value=24.0,
            value=8.0,
            step=0.5,
            help="Matches the FastAPI validation: 6 to 24 hours."
        )

    with col2:
        study_hours = st.number_input(
            "Study Hours / Day",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

    with col3:
        physical_activity_hours = st.number_input(
            "Physical Activity Hours / Day",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.25
        )

    submitted = st.form_submit_button(
        "🔮 Predict Mental Health Score",
        use_container_width=True,
        type="primary"
    )


if submitted:
    country_group = country if country in TOP_COUNTRIES else "Other"

    input_row = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Academic_Level": academic_level,
        "Most_Used_Platform": most_used_platform,
        "Purpose_Of_Use": purpose_of_use,
        "Avg_Daily_Usage_Hours": avg_daily_usage_hours,
        "Daily_Unlocks": daily_unlocks,
        "Study_Hours": study_hours,
        "Physical_Activity_Hours": physical_activity_hours,
        "Sleep_Hours_Per_Night": sleep_hours,
        "Stress_Level": stress_level,
        "Grouped_country": country_group,
    }])

    try:
        prediction = float(model.predict(input_row)[0])

        st.divider()
        st.subheader("📈 Prediction")

        st.markdown(
            f"""
            <div class="result">
                <div class="muted">Predicted Mental Health Score</div>
                <div class="score">{prediction:.2f}</div>
                <div class="muted">Generated by your trained ML model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "This score is a machine-learning prediction, not a medical diagnosis. "
            "Use it as a project/demo output rather than a clinical assessment."
        )

        with st.expander("View model input"):
            st.dataframe(input_row, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption(
            "Make sure the model was trained with the same feature names, "
            "categorical values, and preprocessing expected by the FastAPI code."
        )

st.divider()
st.caption("Student Mental Health Predictor • Streamlit frontend for your ML model")
