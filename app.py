import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the saved CNN model
model = tf.keras.models.load_model('food_model.h5')

# Define the class names (Update this list to match your actual labels)
class_names = ['dhokla', 'chapati', 'kadai_paneer', 'masala_dosa', 'jalebi', 'chai', 'pakode', 'butter_naan', 'samosa', 'idli', 'kaathi_rolls', 'paani_puri', 'fried_rice', 'kulfi', 'momos', 'dal_makhani', 'pav_bhaji', 'burger', 'pizza', 'chole_bhature']  # 🛠 Replace with your own class names

# Streamlit page settings
st.set_page_config(page_title="🍽 Food Classifier", layout="centered")
st.title("🍕 Food Image Classifier using CNN")
st.write("Upload a food image and the model will predict what it is.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # ✅ Use the input shape your model was trained with
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.success(f"🍱 Predicted: *{predicted_class}*")
    st.info(f"📊 Confidence: {confidence:.2f}%")