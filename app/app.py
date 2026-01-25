import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from model.model import predict_disease
from PIL import Image

st.set_page_config(page_title="FarmView", page_icon="🌱")

st.title("🌱 FarmView")
st.subheader("Diagnostic intelligent des maladies des plantes")

uploaded_file = st.file_uploader(
    "📷 Téléversez une image de feuille",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    if st.button("Analyser la feuille"):
        with st.spinner("Analyse en cours..."):
            os.makedirs("temp", exist_ok=True)
            image_path = os.path.join("temp", uploaded_file.name)
            image.save(image_path)

            resultat = predict_disease(image_path)

    st.success(f"🦠 Maladie détectée : {resultat['maladie']}")
    st.write(f"📖 **Description** : {resultat['description']}")
    st.write(f"⚠️ **Niveau de gravité** : {resultat['niveau']}")
    st.write(f"💡 **Conseils** : {resultat['conseils']}")


       
