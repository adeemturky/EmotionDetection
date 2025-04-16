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
This Emotion Detection App supports:
- 🇬🇧 English model (TF-IDF + SVM)
- 🇸🇦 Arabic model (TF-IDF + SVM)

Choose your language, type a sentence, and the model will predict if it's 😊 Happy or 😢 Sad.

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

    .title {text-align: center; font-size: 38px; font-weight: bold; color: #2DC64A;}
    .subtitle {text-align: center; font-size: 20px; color: #888;}
    .result {
        border-radius: 10px;
        padding: 16px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .happy {background-color: #2DC64A;}
    .sad {background-color: #C63C2D;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🎭 Emotion Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-language support: English & Arabic 🌍</div>', unsafe_allow_html=True)
st.write("")

# -----------------------------
# Inputs
# -----------------------------
language = st.selectbox("🌐 Choose language:", ["English", "Arabic"])
text = st.text_area("📝 Enter your sentence:" if language == "English" else "📝 أدخل الجملة:", height=100)
predict_btn = st.button("🎯 Predict" if language == "English" else "🎯 توقّع")

# -----------------------------
# Load model based on language
# -----------------------------
if language == "English":
    model = joblib.load("models/svm_pipeline.joblib")
else:
    model = joblib.load("models/svm_pipeline_ar.joblib")

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_btn:
    if not text.strip():
        st.warning("Please enter some text." if language == "English" else "رجاءً أدخل نصًا.")
    else:
        with st.spinner("Analyzing..." if language == "English" else "جارٍ التحليل..."):
            pred = model.predict([text])[0]
            proba = model.predict_proba([text])[0]
            confidence = round(np.max(proba) * 100, 2)

        if pred == 1:
            st.markdown(f'<div class="result happy">😊 '
                        f'Emotion: Happy<br>Confidence: {confidence}%</div>'
                        if language == "English" else
                        f'<div class="result happy">😊 العاطفة: سعيدة<br>الدقة: {confidence}%</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result sad">😢 '
                        f'Emotion: Sad<br>Confidence: {confidence}%</div>'
                        if language == "English" else
                        f'<div class="result sad">😢 العاطفة: حزينة<br>الدقة: {confidence}%</div>',
                        unsafe_allow_html=True)
