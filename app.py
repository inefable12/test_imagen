import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ======================
# Cargar modelo TM
# ======================
#@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/keras_model.h5", compile=False)
    return model

model = load_model()

# Ajusta estas clases según tu modelo de Teachable Machine
CLASS_NAMES = ["vaca", "bartolito", "percheron"]

st.title("🎥 Clasificación en vivo con Teachable Machine")

# ======================
# Cámara en Streamlit
# ======================
st.markdown("Activa tu cámara y el modelo reconocerá en tiempo real.")

camera_input = st.camera_input("Tomar una foto")

if camera_input:
    # Procesar la imagen
    img = Image.open(camera_input)
    img = img.convert("RGB")
    img_resized = img.resize((224, 224))  # tamaño de Teachable Machine
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    # Mostrar resultados
    st.image(img, caption="Imagen capturada", use_container_width=True)
    st.success(f"Predicción: **{CLASS_NAMES[class_idx]}** ({confidence*100:.2f}%)")
