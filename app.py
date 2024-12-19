import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/model.keras')

model = load_model()

st.title("Cinsiyet Tahmini Uygulaması")
st.write("Model, yüklediğiniz resimdeki kişinin cinsiyetini tahmin eder.")

uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Yüklenen Resim", use_container_width=True)

    img = img.resize((250, 250))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array, verbose=0)

    if prediction[0] > 0.5:
        result = f"Modelin tahmini: Erkek ({prediction[0][0]:.2f})"
    else:
        result = f"Modelin tahmini: Kadın ({1 - prediction[0][0]:.2f})"

    st.success(result)
