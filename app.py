import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSense Pakistan",
    page_icon="🌾",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(160deg, #0a1a0f 0%, #0f2318 50%, #0a1a0f 100%);
    color: #e8f5e9;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(100, 200, 120, 0.15);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #7dda8a;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1.1;
}
.hero p {
    color: #6aab76;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4caf60;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(45, 106, 79, 0.35), rgba(27, 67, 50, 0.5));
    border: 1px solid rgba(125, 218, 138, 0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    backdrop-filter: blur(10px);
}
.result-crop {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #7dda8a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0;
}
.result-confidence {
    font-size: 0.9rem;
    color: #6aab76;
    margin-top: 0.25rem;
    font-weight: 300;
}
.result-divider {
    border: none;
    border-top: 1px solid rgba(125, 218, 138, 0.2);
    margin: 1.25rem 0;
}

/* Top 3 crops */
.crop-bar-label {
    font-size: 0.85rem;
    color: #b2dfb8;
    margin-bottom: 0.2rem;
}
.crop-bar-outer {
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    height: 8px;
    margin-bottom: 0.7rem;
    overflow: hidden;
}
.crop-bar-inner {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4caf60, #7dda8a);
}

/* Alert boxes */
.alert-warn {
    background: rgba(255, 193, 7, 0.1);
    border-left: 3px solid #ffc107;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #ffe082;
}
.alert-ok {
    background: rgba(76, 175, 96, 0.1);
    border-left: 3px solid #4caf60;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #a5d6a7;
}

/* Slider labels */
[data-testid="stSlider"] label {
    color: #b2dfb8 !important;
    font-size: 0.88rem !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2d6a4f, #40916c);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.85rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    cursor: pointer;
    margin-top: 1rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #40916c, #52b788);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(64, 145, 108, 0.4);
}

/* Footer */
.footer {
    text-align: center;
    color: #2d6a4f;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(100, 200, 120, 0.1);
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('agrisense_crop_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌾 AgriSense</h1>
    <p>PRECISION AGRICULTURE · PAKISTAN · AI-POWERED CROP INTELLIGENCE</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model file not found. Make sure `agrisense_crop_model.pkl` is in the same folder as `app.py`.")
    st.stop()

# ── Sensor inputs ───────────────────────────────────────────────
st.markdown('<div class="section-label">🧪 Soil Nutrients (NPK Sensor)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    N = st.slider("Nitrogen (N)", 0, 140, 90, help="Soil nitrogen level")
with col2:
    P = st.slider("Phosphorus (P)", 5, 145, 42, help="Soil phosphorus level")
with col3:
    K = st.slider("Potassium (K)", 5, 205, 43, help="Soil potassium level")

st.markdown('<div class="section-label">🌤️ Weather Conditions (Weather Station)</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.5, step=0.5)
with col5:
    humidity = st.slider("Humidity (%)", 14.0, 100.0, 80.0, step=0.5)
with col6:
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 200.0, step=5.0)

st.markdown('<div class="section-label">🌱 Soil Quality (Soil Sensor)</div>', unsafe_allow_html=True)
ph = st.slider("Soil pH", 3.5, 9.9, 6.5, step=0.1)

# ── Predict button ──────────────────────────────────────────────
predict = st.button("⚡ Analyse Farm & Predict Crop")

# ── Prediction output ───────────────────────────────────────────
if predict:
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                               columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3 = [(model.classes_[i], probabilities[i] * 100) for i in top3_idx]

    # Main result card
    top3_bars_html = ""
    for crop, prob in top3:
        top3_bars_html += f"""
        <div class="crop-bar-label">{crop.capitalize()} — {prob:.1f}%</div>
        <div class="crop-bar-outer">
            <div class="crop-bar-inner" style="width:{prob}%"></div>
        </div>
        """

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:0.7rem; letter-spacing:0.15em; color:#4caf60; text-transform:uppercase; font-weight:700;">
            ✅ Best Crop Recommendation
        </div>
        <div class="result-crop">{prediction}</div>
        <div class="result-confidence">Model confidence: {confidence:.1f}%</div>
        <hr class="result-divider">
        <div style="font-size:0.7rem; letter-spacing:0.12em; color:#4caf60; text-transform:uppercase; font-weight:700; margin-bottom:0.75rem;">
            Top 3 Options
        </div>
        {top3_bars_html}
    </div>
    """, unsafe_allow_html=True)

    # Actionable instructions
    st.markdown('<div class="section-label" style="margin-top:1.5rem">📋 AgriSense Recommended Actions</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="alert-ok">1. Prepare your soil for <strong>{prediction.upper()}</strong> cultivation this season.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-ok">2. Current NPK ratio — N: {N} | P: {P} | K: {K}</div>', unsafe_allow_html=True)

    if humidity > 85:
        st.markdown(f'<div class="alert-warn">⚠️ High humidity ({humidity}%) — monitor crops closely for fungal disease. Consider antifungal spray.</div>', unsafe_allow_html=True)
    elif humidity < 40:
        st.markdown(f'<div class="alert-warn">⚠️ Low humidity ({humidity}%) — increase irrigation frequency to avoid crop stress.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">3. ✅ Humidity is at a healthy level ({humidity}%) — no irrigation changes needed.</div>', unsafe_allow_html=True)

    if ph < 5.5:
        st.markdown(f'<div class="alert-warn">⚠️ Soil is too acidic (pH {ph}) — apply agricultural lime to raise pH above 5.5.</div>', unsafe_allow_html=True)
    elif ph > 7.5:
        st.markdown(f'<div class="alert-warn">⚠️ Soil is too alkaline (pH {ph}) — apply elemental sulfur to lower pH below 7.5.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">4. ✅ Soil pH is healthy ({ph}) — ideal for most crops.</div>', unsafe_allow_html=True)

    if temperature > 38:
        st.markdown(f'<div class="alert-warn">⚠️ Very high temperature ({temperature}°C) — consider shade nets or evening irrigation to protect crops.</div>', unsafe_allow_html=True)
    elif temperature < 12:
        st.markdown(f'<div class="alert-warn">⚠️ Low temperature ({temperature}°C) — frost risk present. Consider protective coverings for seedlings.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">5. ✅ Temperature is within a healthy range ({temperature}°C).</div>', unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AGRISENSE PAKISTAN · SOWING DATA, HARVESTING THE FUTURE 🌾<br>
    PAK-AUSTRIA FACHHOCHSCHULE · ENTREPRENEURSHIP PROJECT 2025
</div>
""", unsafe_allow_html=True)
