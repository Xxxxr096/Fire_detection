import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Charger le modèle de détection d'incendie
model = load_model("FIRE_DETECTION.h5")


# Fonction pour prétraiter l'image
def preprocess_image(image):
    image = image.resize((150, 150))  # Redimensionner l'image
    image = np.array(image) / 255.0  # Normaliser l'image
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le modèle
    return image


# Fonction pour détecter l'incendie
def detect_fire(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return (
        prediction[0][0] < 0.5
    )  # Si la prédiction est supérieure à 0.5, il y a un incendie


# Interface Streamlit
st.title("Détection d'Incendie")
st.write("Téléchargez une image pour détecter si elle contient un incendie.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée.", use_column_width=True)
    st.write("")
    st.write("Analyse en cours...")

    if detect_fire(image):
        st.error("🔥 Incendie détecté ! 🔥")
    else:
        st.success("✅ Aucun incendie détecté.")
