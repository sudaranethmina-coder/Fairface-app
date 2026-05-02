import streamlit as st
import torch
from PIL import Image
from collections import Counter
import pandas as pd
import altair as alt
from MLModel import CNN, eval_transform

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="FairVision AI",
    layout="wide",
    page_icon="🧠"
)

# =========================
# STYLING (COLORS)
# =========================
st.markdown("""
<style>
    .title {
        text-align:center;
        font-size:42px;
        font-weight:bold;
        color:#4A90E2;
    }
    .card {
        background-color:#111827;
        padding:10px;
        border-radius:12px;
        text-align:center;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LABELS
# =========================
ages = [
    "0-2","3-9","10-19","20-29","30-39",
    "40-49","50-59","60-69","70+"
]

# =========================
# MODEL
# =========================
model = CNN(len(ages))
checkpoint = torch.load("best_fairface_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = eval_transform

# =========================
# HEADER
# =========================
st.markdown('<div class="title">🧠 FairVision Age Classifier</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📤 Upload Face Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================
# STORAGE FOR CHART
# =========================
predictions = []

# =========================
# GRID DISPLAY
# =========================
if uploaded_files:

    cols = st.columns(3)

    for idx, file in enumerate(uploaded_files):

        img = Image.open(file).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            predicted_age = ages[pred_class]

        predictions.append(predicted_age)

        col = cols[idx % 3]

        with col:
            st.image(img, use_container_width=True)
            st.markdown(
                f"<div class='card'>"
                f"<h3 style='color:#00C853;'>🎯 {predicted_age}</h3>"
                f"<p style='color:#90CAF9;'>Confidence: {confidence:.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

# =========================
# 📊 LINE CHART (ONLY IF MULTIPLE IMAGES)
# =========================
if uploaded_files and len(uploaded_files) > 1:

    st.markdown("## 📊 Age Distribution Analysis")

    count = Counter(predictions)

    df = pd.DataFrame({
        "Age Group": list(count.keys()),
        "Count": list(count.values())
    })

    chart = alt.Chart(df).mark_line(point=True).encode(
        x="Age Group",
        y="Count"
    ).properties(width=700)

    st.altair_chart(chart, use_container_width=True)