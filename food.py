import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the model
model = tf.keras.models.load_model('food_model.h5')

import json

# Load class names dynamically from saved indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary to map index to label
class_names = {v: k for k, v in class_indices.items()}
# Load class names (preferably from a JSON file if saved during training)
# class_names = ['dhokla', 'chapati', 'kadai_paneer', 'masala_dosa', 'jalebi', 'chai', 'pakode', 
#                'butter_naan', 'samosa', 'idli', 'kaathi_rolls', 'paani_puri', 'fried_rice', 
#                'kulfi', 'momos', 'dal_makhani', 'pav_bhaji', 'burger', 'pizza', 'chole_bhature']  # Ensure this matches the training order

# Streamlit settings
st.set_page_config(page_title="🍽 Food Classifier", layout="centered")
st.title("🍕 Food Image Classifier using CNN")
st.write("Upload a food image and the model will predict what it is.")

# Upload image
uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # MobileNetV2 expects 224x224
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # ✅ CORRECT preprocessing for MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display result
    st.success(f"🍱 Predicted: {predicted_class}")
    st.info(f"📊 Confidence: {confidence:.2f}%")