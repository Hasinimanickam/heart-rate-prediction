import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_rate_model.pkl")

st.set_page_config(
    page_title="Heart Rate AI",
    page_icon="❤️",
    layout="centered"
)
# Sidebar
with st.sidebar:
    st.title("❤️ Heart Rate AI")
    st.write("Your ML Prediction Assistant")

    st.divider()

    st.markdown("### 🏠 Home")
    st.markdown("### ❤️ Prediction")
    st.markdown("### ℹ️ About")

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
   
    background: linear-gradient(135deg, #ffe4e6, #ede9fe, #e0f2fe);

}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

/* Main title */
.hero {
    text-align: center;
    padding: 25px 10px;
}

.hero-icon {
    font-size: 55px;
    animation: heartbeat 1.3s infinite;
}

@keyframes heartbeat {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.15);
    }
}


.hero h1 {
    font-size: 44px;
    margin: 5px 0;
    font-weight: 800;
    background: linear-gradient(90deg, #e63946, #8e44ad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #666;
    font-size: 17px;
}

/* Input card */
.input-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    margin-top: 20px;
}
.stNumberInput input {
    border-radius: 12px;
    border: 2px solid #e9d5ff;
    padding: 10px;
}

.stNumberInput input:focus {
    border-color: #8e44ad;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 15px;
    height: 55px;
    font-size: 18px;
    font-weight: 700;
    border: none;
    background: linear-gradient(90deg, #e63946, #8e44ad);
    color: white;
    box-shadow: 0 6px 20px rgba(142, 68, 173, 0.25);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(142, 68, 173, 0.35);
}

}

/* Result */
.result-card {
    background: linear-gradient(135deg, #e63946, #8e44ad);
    color: white;
    padding: 35px;
    border-radius: 25px;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.18);
}

.result-card .bpm {
    font-size: 58px;
    font-weight: 800;
    margin: 10px 0;
}

.result-card p {
    font-size: 16px;
}
.result-card {
    animation: fadeInUp 0.7s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.step-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    transition: 0.3s;
}

.step-card:hover {
    transform: translateY(-5px);
}

.step-icon {
    font-size: 35px;
    margin-bottom: 8px;
}

.step-card h3 {
    margin: 5px 0;
}

.step-card p {
    color: #666;
    font-size: 14px;
}
/* Footer */
.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)



# ---------------- Hero Section ----------------

st.markdown("""
<div class="hero">
    <div class="hero-icon">❤️</div>
    <h1>Heart Rate AI</h1>
    <p>Machine Learning Based Maximum Heart Rate Prediction</p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("🔍 How It Works")

step1, step2 = st.columns(2)

with step1:
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">📝</div>
        <h3>Enter Details</h3>
        <p>Provide your age, sleep, exercise and heart-rate values.</p>
    </div>
    """, unsafe_allow_html=True)

with step2:
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">🤖</div>
        <h3>ML Processing</h3>
        <p>The trained machine-learning model processes your inputs.</p>
    </div>
    """, unsafe_allow_html=True)

step3, step4 = st.columns(2)

with step3:
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">🔮</div>
        <h3>Predict</h3>
        <p>The model estimates your maximum heart rate.</p>
    </div>
    """, unsafe_allow_html=True)

with step4:
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">📊</div>
        <h3>View Result</h3>
        <p>Your prediction is displayed instantly.</p>
    </div>
    """, unsafe_allow_html=True)




# ---------------- Input Section ----------------

st.markdown("""
<div class="input-card">
<h3>📝 Enter Your Details</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "👤 Age",
        min_value=10,
        max_value=100,
        value=25
    )

    sleep = st.number_input(
        "😴 Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=8.0
    )

    exercise = st.number_input(
        "🏃 Exercise Days / Week",
        min_value=0,
        max_value=7,
        value=3
    )

with col2:

    rest_before = st.number_input(
        "❤️ Resting Heart Rate Before",
        min_value=30,
        max_value=150,
        value=70
    )

    rest_after = st.number_input(
        "❤️ Resting Heart Rate After",
        min_value=30,
        max_value=150,
        value=65
    )


st.write("")
# ---------------- Prediction ----------------

if st.button("🔮 Predict Maximum Heart Rate"):

    input_data = pd.DataFrame([[
        age,
        sleep,
        exercise,
        rest_before,
        rest_after
    ]], columns=[
        "Age",
        "Sleep Hours",
        "Exercise Frequency (Days/Week)",
        "Resting Heart Rate Before",
        "Resting Heart Rate After"
    ])

    prediction = model.predict(input_data)[0]

    # Result card
    st.markdown(
        f"""
        <div class="result-card">
            <h2>🎯 Prediction Result</h2>
            <div class="bpm">{prediction:.0f} BPM</div>
            <p>Predicted Maximum Heart Rate</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Details
    st.subheader("📋 Your Input Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Age", f"{age}")

    with c2:
        st.metric("Sleep", f"{sleep} hrs")

    with c3:
        st.metric("Exercise", f"{exercise} days/week")

    st.progress(
        min(int(prediction / 200 * 100), 100),
        text="Prediction level"
    )

    st.info(
        "💡 This result is generated by a machine-learning model "
        "based on the information you entered."
    )
st.caption("⚠️ This prediction is for educational purposes only and is not medical advice.")
st.success("🌟 Prediction completed successfully!")

