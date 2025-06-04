import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('fruit_classification_model.h5')

# Class labels (get from your training set)
class_labels = ['Apple', 'Banana', 'Dates', 'Guava', 'Jujube']  # <- Replace with your actual class names

# Set page config
st.set_page_config(page_title="Fruit Image Classifier", layout="centered")

# Title
st.title("🍎 Fruit Image Classifier")
st.write("Upload an image of a fruit, and I will tell you what it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_image(img):
    img = img.resize((150, 150))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    return class_labels[class_index], confidence

# If an image is uploaded
if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)

    # Predict
    with st.spinner("Classifying..."):
        predicted_class, confidence = predict_image(image_display)
    
    # Display results
    st.success(f"Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
