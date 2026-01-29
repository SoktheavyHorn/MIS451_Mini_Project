import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

# Load labels
with open("labels.txt", "r") as f:
    class_names = f.read().splitlines()

st.set_page_config(page_title="Image Classifier")
st.title("🖼️ Teachable Machine – Image Upload")

st.write("Upload an image and get a prediction.")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess (Teachable Machine standard)
    image = image.resize((224, 224))
    image_array = np.asarray(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    index = np.argmax(predictions)
    confidence = float(predictions[0][index])

    st.success(f"Prediction: **{class_names[index]}**")
    st.write(f"Confidence: `{confidence:.2f}`")
