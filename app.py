import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/model.keras')

model = load_model()

# Streamlit app
st.title("Cinsiyet Tahmini Uygulaması")
st.write("Model, yüklediğiniz resimdeki kişinin cinsiyetini tahmin eder.")

uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Yüklenen Resim", use_column_width=True)

    # Preprocess the image
    img = img.resize((250, 250))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    # Interpret the result
    if prediction[0] > 0.5:
        result = f"Modelin tahmini: Erkek ({prediction[0][0]:.2f})"
    else:
        result = f"Modelin tahmini: Kadın ({1 - prediction[0][0]:.2f})"

    # Display the result
    st.write(result)
