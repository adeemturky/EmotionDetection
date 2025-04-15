import streamlit as st
import joblib
import numpy as np
import time

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

# -----------------------------
# Sidebar - About 🌍
# -----------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This Emotion Detection App uses a machine learning model (TF-IDF + SVM)
to predict the emotion in your text.

Just type your sentence and the app will tell you if it's **😊 Happy** or **😢 Sad** — with confidence!

**Created by Adeem 💚**
""")

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #FFFFFF;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #888;
    }

    .result {
        border-radius: 10px;
        padding: 14px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .happy {
        background-color: #2DC64A;
    }

    .sad {
        background-color: #C63C2D;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🎭 Emotion Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a sentence and get the emotion prediction ✨</div>', unsafe_allow_html=True)
st.write("")

# -----------------------------
# Input & Button
# -----------------------------
text = st.text_area("📝 Enter your sentence:", height=100)
predict_btn = st.button("🎯 Predict")

# -----------------------------
# Load model
# -----------------------------
svm_pipeline = joblib.load("emotion-detector/models/svm_pipeline.joblib")

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_btn:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        status_placeholder = st.empty()
        status_placeholder.info("🤖 Analyzing your sentence...")
        time.sleep(1.2)

        pred = svm_pipeline.predict([text])[0]
        proba = svm_pipeline.predict_proba([text])[0]
        confidence = round(np.max(proba) * 100, 2)

        status_placeholder.empty()

        if pred == 1:
            st.markdown(f'<div class="result happy">😊 Emotion: Happy<br>🧠 Confidence: {confidence}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result sad">😢 Emotion: Sad<br>🧠 Confidence: {confidence}%</div>', unsafe_allow_html=True)
