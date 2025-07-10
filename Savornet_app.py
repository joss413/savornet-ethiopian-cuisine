import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import os
import matplotlib.pyplot as plt
from PIL import Image

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

# Model loader
@st.cache_resource
def load_models():
    dense_path = "dense_final_model.h5"
    resnet_path = "resnet50v2_final4_model.h5"
    savornet_path = "singletest_attention_model.h5"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    dense_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{dense_path}"
    resnet_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{resnet_path}"
    savornet_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{savornet_path}"

    try:
        for path, url, name in [
            (dense_path, dense_url, "DenseNet"),
            (resnet_path, resnet_url, "ResNet"),
            (savornet_path, savornet_url, "SavorNet"),
        ]:
            if not os.path.exists(path):
                st.info(f"🔽 Downloading {name} model from Hugging Face...")
                with requests.get(url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

        densenet = tf.keras.models.load_model(dense_path)
        resnet = tf.keras.models.load_model(resnet_path)
        savornet = tf.keras.models.load_model(savornet_path)

        return densenet, resnet, savornet

    except Exception as e:
        st.error(f"❌ Failed to load models from Hugging Face: {e}")
        st.stop()

# Preprocessing
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = array / 255.0
    return np.expand_dims(array, axis=0)

# Top-K predictions
def get_top_k(preds, k=2):
    top_indices = np.argsort(preds)[::-1][:k]
    return [(CLASS_NAMES[i], preds[i]) for i in top_indices]

# Load models
densenet, resnet, savornet = load_models()

# Sidebar Info
st.sidebar.title("ℹ️ Info")
st.sidebar.markdown("""
**Models Used**
- **DenseNet121**
- **ResNet50V2**
- **SavorNet (Attention Fusion)**

**Test Accuracies**
- DenseNet121: 83.12%
- ResNet50V2: 86.36%
- Ensemble: 88.31%
- **SavorNet**: **92.20%**
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

    # Show results
    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Ensemble Model Prediction:** {pred_label} ({pred_ensemble[top1_idx]*100:.2f}%)")

    st.subheader("🧠 SavorNet (Adaptive Attention) Prediction (Top-2)")
    for name, prob in get_top_k(pred_savornet):
        st.write(f"**{name}**: {prob:.2%}")

    st.subheader("📊 Confidence Comparison")
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

    axs[0].barh(CLASS_NAMES, pred_dense, color="skyblue")
    axs[0].invert_yaxis()
    axs[0].set_title("DenseNet121")

    axs[1].barh(CLASS_NAMES, pred_res, color="orange")
    axs[1].invert_yaxis()
    axs[1].set_title("ResNet50V2")

    axs[2].barh(CLASS_NAMES, pred_ensemble, color="green")
    axs[2].invert_yaxis()
    axs[2].set_title("Soft Voting Ensemble")

    axs[3].barh(CLASS_NAMES, pred_savornet, color="purple")
    axs[3].invert_yaxis()
    axs[3].set_title("SavorNet (Attention)")

    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")

    st.pyplot(fig)

    st.markdown(f"""
    ### 📈 Model Performance
    | Model        | Test Accuracy | Top-1 Confidence |
    |--------------|---------------|------------------|
    | DenseNet121  | {DENSENET_ACC:.2f}%       | {pred_dense[top1_idx]*100:.2f}%          |
    | ResNet50V2   | {RESNET_ACC:.2f}%       | {pred_res[top1_idx]*100:.2f}%          |
    | Ensemble     | {ENSEMBLE_ACC:.2f}%     | {pred_ensemble[top1_idx]*100:.2f}%      |
    | **SavorNet** | **{SAVORNET_ACC:.2f}%** | **{pred_savornet[top1_idx]*100:.2f}%**  |
    """)

else:
    st.info("📤 Upload an image to start classification.")

# About Section
st.markdown("""
---

## 📚 About

**SavorNet** is a deep learning-based Ethiopian food classification system. It uses:
- **DenseNet121**
- **ResNet50V2**
- **SavorNet**: a novel attention-based fusion model

**SavorNet** adaptively fuses features from both CNN backbones using a two-layer attention mechanism, achieving **92.20% accuracy** and **96.10% top-2 accuracy**.

### 🏆 Accuracies:
- DenseNet121: 83.12%
- ResNet50V2: 86.36%
- Ensemble (Soft Voting): 88.31%
- **SavorNet (Attention)**: **92.20%**

**Author:** Yoseph Negash  
**Contact:** yosephn22@gmail.com  
**Year:** 2025
""")
