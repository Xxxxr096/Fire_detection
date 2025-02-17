import streamlit as st
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


# Configuration du style Streamlit
st.set_page_config(page_title="Détection d'Incendie", page_icon="🔥", layout="centered")

# Ajout de CSS personnalisé pour la police et la couleur de fond
st.markdown(
    """
    <style>
        body {
            background-color: #f4f7fa;  /* Couleur de fond personnalisée */
            font-family: 'Verdana', sans-serif;  /* Changer la police à Verdana */
        }
        .stTitle {
            color: #ff5722;  /* Couleur du titre */
            font-size: 36px;
            font-weight: bold;
            text-align: center;
        }
        .stText {
            font-size: 18px;
            color: #555555;
            text-align: center;
        }
        .stFileUploader {
            background-color: #ff9800;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stButton {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .css-1h6y2s0 {
            background-color: #ff5722 !important;
            color: white !important;
            border-radius: 10px;
            padding: 20px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .css-1v3vjjj {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 10px;
            padding: 20px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Interface Streamlit améliorée
st.title("Détection d'Incendie")
st.write("Téléchargez une image pour détecter si elle contient un incendie.")
st.markdown("<br>", unsafe_allow_html=True)

# Ajout d'un upload de fichier
uploaded_file = st.file_uploader(
    "Choisissez une image...", type=["jpg", "jpeg", "png"], key="file_uploader"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Image téléchargée.",
        use_container_width=True,
        output_format="PNG",
    )  # Remplacer use_column_width par use_container_width
    st.write("")

    # Ajout d'une animation en attendant la détection
    st.write("🔍 Analyse en cours...")

    # Détection d'incendie
    if detect_fire(image):
        st.error("🔥 Incendie détecté ! 🔥", icon="🚨")  # Retirer class_name
    else:
        st.success("✅ Aucun incendie détecté.", icon="✅")  # Retirer class_name
