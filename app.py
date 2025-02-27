import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.backend import clear_session
import os

# 🛠 **Force TensorFlow to use CPU-only mode**
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

# 🔥 **Reduce precision to float16 for better CPU performance**
tf.keras.backend.set_floatx("float16")

# 🚀 **Free up TensorFlow memory before loading models**
clear_session()

# 🌟 **Streamlit Page Configuration**
st.set_page_config(page_title="Eye Disease Classifier", layout="wide")

# 🎨 **Custom UI Styling**
st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    .stApp { background-color: #0E1117; }
    .big-font { font-size:24px !important; font-weight: bold; color: #4CAF50; }
    .small-font { font-size:14px !important; color: #aaaaaa; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: white; }
    </style>
    """, unsafe_allow_html=True)

# 📌 **Function to Load Models Safely**
def load_model_safe(model_path):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading {model_path}: {e}")
        return None

# 🔄 **Load Models**
model1 = load_model_safe("model2.h5")        # Model 1 expects (224, 224)
model2 = load_model_safe("model1.keras")     # Model 2 expects (256, 256)

# 🧠 **Load Meta-Classifier**
try:
    with open("meta_model.pkl", "rb") as f:
        meta_model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading meta_model.pkl: {e}")
    meta_model = None

# 🏷 **Disease Labels**
disease_labels = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Macular Degeneration", "Normal"]

# 🖼 **Function to Preprocess Image (Dynamic Resizing Based on Model)**
def preprocess_image(img, target_size):
    img_resized = img.resize(target_size)  # Resize using PIL
    img_array = img_to_array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# 🔥 **Streamlit UI - Header**
st.markdown("<h1 style='text-align: center; color: #333;'>👁️ Eye Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='big-font' style='text-align: center;'>Upload a Retinal Image for Disease Prediction</p>", unsafe_allow_html=True)
st.write("---")

# 📤 **File Upload**
uploaded_file = st.file_uploader("📎 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 📷 **Display Uploaded Image**
    st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

    # Load the image without resizing
    image = load_img(uploaded_file)

    # ✅ **Check if models are loaded before making predictions**
    if model1 and model2 and meta_model:
        try:
            # 🏥 **Get Predictions from Both Models**
            pred1 = model1.predict(preprocess_image(image, (224, 224))).flatten()
            pred2 = model2.predict(preprocess_image(image, (256, 256))).flatten()

            # 📝 **Log Predictions for Debugging**
            st.write(f"🔍 Model 1 Prediction: {pred1}")
            st.write(f"🔍 Model 2 Prediction: {pred2}")

            # 🔗 **Combine Model Predictions**
            combined_pred = np.concatenate([pred1, pred2])

            # 📏 **Ensure Feature Vector Matches Expected Input Size (9 Features)**
            if len(combined_pred) != 9:
                if len(combined_pred) > 9:
                    combined_pred = combined_pred[:9]  # Trim extra features
                else:
                    combined_pred = np.pad(combined_pred, (0, 9 - len(combined_pred)), mode="constant")  # Pad missing features

            # 🤖 **Meta-Classifier Prediction**
            final_pred = meta_model.predict([combined_pred])[0]
            disease_name = disease_labels[final_pred]

            # 🎯 **Show Final Prediction**
            st.markdown(f"<p class='big-font' style='text-align: center; color: #ff5733;'>🔍 Prediction: {disease_name}</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

# 🚀 **Footer**
st.markdown("---")
st.markdown("<p class='small-font' style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
