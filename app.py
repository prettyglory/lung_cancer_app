import streamlit as st # type: ignore
from PIL import Image
import torch
from model import SimpleCNN
from utils import predict_image, device

# Load model
model = SimpleCNN(num_classes=3).to(device)
model.load_state_dict(torch.load("lung_cancer_model.pth", map_location=device))
model.eval()

class_names = ['adenocarcinoma', 'benign', 'squamous_cell_carcinoma']
threshold = 0.7

# Streamlit interface
st.set_page_config(page_title="Lung Cancer Classifier", layout="wide")
st.title("Lung Cancer Histopathology Classifier")
st.write("Upload an image. The model will classify or reject unrelated images.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    result = predict_image(model, image, class_names, threshold)
    st.success(result)
