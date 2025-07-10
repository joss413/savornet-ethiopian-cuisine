import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import os
import matplotlib.pyplot as plt
from PIL import Image
import zipfile

# App Configuration
st.set_page_config(page_title="SavorNet: Ethiopian Food Classifier", layout="wide")
st.title("🍲 SavorNet: Ethiopian Cuisine Classifier")
st.markdown("Classifies traditional Ethiopian dishes using deep learning (DenseNet121, ResNet50V2, SavorNet).")

# Constants
HF_TOKEN = st.secrets["hf_token"]
REPO_ID = "Jossi18/Ethiopian_cusine_classification"
IMG_SIZE = 512
CLASS_NAMES = [
    "Beyaynet", "Chechebsa", "Doro Wot", "Dulet", "Enkulal Firfir",
    "Firfir", "Genfo", "Injera", "Kik Alicha", "Shiro", "Tibs"
]

# Accuracy values
DENSENET_ACC = 83.12
RESNET_ACC = 86.36
ENSEMBLE_ACC = 88.31
SAVORNET_ACC = 92.20
SAVORNET_TOP2 = 96.10

# Cached Model Loader
@st.cache_resource
def load_models():
    dense_path = "dense_final_model.h5"
    resnet_path = "resnet50v2_final4_model.h5"
    savornet_zip = "singletest_attention_model_tf.zip"
    savornet_dir = "singletest_attention_model_tf"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    dense_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{dense_path}"
    resnet_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{resnet_path}"
    savornet_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{savornet_zip}"

    try:
        # Download DenseNet
        if not os.path.exists(dense_path):
            st.info("🔽 Downloading DenseNet model...")
            with requests.get(dense_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(dense_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

        # Download ResNet
        if not os.path.exists(resnet_path):
            st.info("🔽 Downloading ResNet model...")
            with requests.get(resnet_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(resnet_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

        # Download and extract SavorNet if missing
        if not os.path.exists(savornet_dir):
            st.info("🔽 Downloading SavorNet model...")
            with requests.get(savornet_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(savornet_zip, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            st.info("📂 Extracting SavorNet...")
            with zipfile.ZipFile(savornet_zip, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(savornet_zip)

        # Load models
        densenet = tf.keras.models.load_model(dense_path)
        resnet = tf.keras.models.load_model(resnet_path)
        savornet = tf.keras.models.load_model(savornet_dir)

        return densenet, resnet, savornet

    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.stop()

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = array / 255.0
    return np.expand_dims(array, axis=0)

# Get top-k predictions
def get_top_k(preds, k=2):
    top_indices = np.argsort(preds)[::-1][:k]
    return [(CLASS_NAMES[i], preds[i]) for i in top_indices]

# Load models once
densenet, resnet, savornet = load_models()

# Sidebar Info
st.sidebar.title("ℹ️ Info")
st.sidebar.markdown(f"""
**Models Used**
- DenseNet121
- ResNet50V2
- SavorNet (Attention Fusion)

**Test Accuracies**
- DenseNet121: {DENSENET_ACC:.2f}%
- ResNet50V2: {RESNET_ACC:.2f}%
- Ensemble: {ENSEMBLE_ACC:.2f}%
- **SavorNet**: **{SAVORNET_ACC:.2f}%**
""")

# Image Upload
uploaded = st.file_uploader("📷 Upload an image of Ethiopian food", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed = preprocess_image(image)

    # Predictions
    pred_dense = densenet.predict(processed)[0]
    pred_res = resnet.predict(processed)[0]
    pred_savornet = savornet.predict(processed)[0]
    pred_ensemble = (pred_dense + pred_res) / 2

    top1_idx = np.argmax(pred_ensemble)
    pred_label = CLASS_NAMES[top1_idx]

    # Result Display
    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Ensemble Prediction:** {pred_label} ({pred_ensemble[top1_idx]*100:.2f}%)")

    st.subheader("🧠 SavorNet (Top-2 Attention Predictions)")
    for name, prob in get_top_k(pred_savornet, k=2):
        st.write(f"**{name}**: {prob:.2%}")

    st.subheader("📊 Confidence Comparison")
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

    axs[0].barh(CLASS_NAMES, pred_dense, color="skyblue")
    axs[0].set_title("DenseNet121")
    axs[1].barh(CLASS_NAMES, pred_res, color="orange")
    axs[1].set_title("ResNet50V2")
    axs[2].barh(CLASS_NAMES, pred_ensemble, color="green")
    axs[2].set_title("Ensemble (Soft Voting)")
    axs[3].barh(CLASS_NAMES, pred_savornet, color="purple")
    axs[3].set_title("SavorNet (Attention)")

    for ax in axs:
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")

    st.pyplot(fig)

    st.markdown(f"""
    ### 📈 Model Performance on This Image
    | Model        | Test Accuracy | Confidence |
    |--------------|---------------|------------|
    | DenseNet121  | {DENSENET_ACC:.2f}%       | {pred_dense[top1_idx]*100:.2f}% |
    | ResNet50V2   | {RESNET_ACC:.2f}%       | {pred_res[top1_idx]*100:.2f}% |
    | Ensemble     | {ENSEMBLE_ACC:.2f}%     | {pred_ensemble[top1_idx]*100:.2f}% |
    | **SavorNet** | **{SAVORNET_ACC:.2f}%** | **{pred_savornet[top1_idx]*100:.2f}%** |
    """)
else:
    st.info("📤 Upload an image to start classification.")

# About Section
st.markdown("""
---

## 📚 About

**SavorNet** is a deep learning-based Ethiopian food classification system using:
- **DenseNet121**
- **ResNet50V2**
- **SavorNet**: an attention-based fusion model

SavorNet adaptively fuses feature maps from both CNNs using a two-layer attention mechanism.

### 🏆 Accuracy Highlights:
- DenseNet121: 83.12%
- ResNet50V2: 86.36%
- Ensemble: 88.31%
- **SavorNet (Attention)**: **92.20%**, Top-2: **96.10%**

**Author:** Yoseph Negash  
**Contact:** yosephn22@gmail.com  
**Year:** 2025
""")
