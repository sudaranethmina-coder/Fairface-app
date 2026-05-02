# =========================================================
# streamlit_app
# =========================================================
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from MLModel import CNN, eval_transform

ages = [
"0-2","3-9","10-19","20-29","30-39",
"40-49","50-59","60-69","70+"
]

model = CNN(len(ages))
model.load_state_dict(torch.load("best_fairface_model.pt"))
model.eval()

transform = eval_transform

st.title("🧠 FairVision Age Classification")

uploaded_file = st.file_uploader("Upload Face Image")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    x = transform(img).unsqueeze(0)
    outputs = model(x)
    st.write(
        outputs
    )