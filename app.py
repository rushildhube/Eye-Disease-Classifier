import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io

# Force TensorFlow to use CPU (Streamlit Cloud has no GPU support)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define model paths
MODEL1_PATH = "model1.keras"
MODEL2_PATH = "model2.h5"
META_MODEL_PATH = "meta_model.pkl"

# Function to load models safely
@st.cache_resource
def load_model_safe(path, model_name):
    try:
        model = tf.keras.models.load_model(path)
        st.success(f"✅ Successfully loaded {model_name}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading {model_name}: {e}")
        return None

# Load models
model1 = load_model_safe(MODEL1_PATH, "Model 1")
model2 = load_model_safe(MODEL2_PATH, "Model 2")

# Load meta model
try:
    with open(META_MODEL_PATH, "rb") as f:
        meta_model = pickle.load(f)
    st.success("✅ Successfully loaded meta_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading meta_model.pkl: {e}")
    meta_model = None

# Function to preprocess images
def preprocess_image(img, model_name):
    """Resize and normalize the image based on the model's input requirements."""
    if model_name == "model1":
        target_size = (256, 256)
    elif model_name == "model2":
        target_size = (224, 224)
    else:
        raise ValueError("Invalid model name")

    img = img.resize(target_size)  # Resize using PIL
    img_array = np.asarray(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.set_page_config(page_title="Eye Disease Classifier", page_icon="👁️", layout="centered")

st.title("👁️ Eye Disease Classification")
st.markdown("### Upload a Retinal Image for Disease Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Extract and display image metadata
    st.subheader("📊 Image Information:")
    st.write(f"**Format:** {image.format}")
    st.write(f"**Size:** {image.size} pixels")

    # Run model predictions
    predictions = {}

    if model1:
        img1 = preprocess_image(image, "model1")
        pred1 = model1.predict(img1).flatten()
        predictions["Model 1"] = pred1
    else:
        pred1 = None

    if model2:
        img2 = preprocess_image(image, "model2")
        pred2 = model2.predict(img2).flatten()
        predictions["Model 2"] = pred2
    else:
        pred2 = None

    # Combine predictions using meta model
    if pred1 is not None and pred2 is not None and meta_model:
        final_pred = meta_model.predict([pred1, pred2])
        confidence = np.max(final_pred) * 100  # Get confidence score

        # Display results
        st.subheader("🩺 Disease Prediction:")
        st.write(f"**Prediction:** {final_pred}")
        st.write(f"**Confidence Score:** {confidence:.2f}%")

        # Download button for prediction results
        result_text = f"Prediction: {final_pred}\nConfidence Score: {confidence:.2f}%"
        result_file = io.BytesIO(result_text.encode())
        st.download_button(label="⬇️ Download Results", data=result_file, file_name="prediction.txt", mime="text/plain")

    else:
        st.error("⚠️ Could not generate prediction. Ensure models are loaded correctly.")

# Footer
st.markdown("---")
st.markdown("👨‍⚕️ **Developed for AI-powered Eye Disease Detection** | 🏥 **Medical AI Project**")
