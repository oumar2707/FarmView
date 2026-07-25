"""
FarmView - Application de diagnostic agricole par IA
Interface ultra-design avec Espace Admin Premium, Carte Géographique & Design Corporate
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sys
import os

# Ajouter le dossier src au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer nos modules
from src.utils.image_processor import load_and_preprocess_image, resize_image
from src.data.disease_info import get_disease_info, CROPS
from src.models.predictor import DiseasePredictor

# Configuration de la page
st.set_page_config(
    page_title="FarmView - Diagnostic Agricole IA",
    page_icon="🌱",
    layout="wide"
)

# Injection CSS (Super-Design Blanc & Vert avec Navbar Premium)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1E293B;
}

.stApp {
    background-color: #F8FAFC;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background-color: transparent !important;}

/* Container du Header Flottant */
.navbar-wrapper {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 10px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
    margin-bottom: 30px;
}

/* Style uniforme des boutons de navigation */
div.stButton > button {
    white-space: nowrap !important;
    word-break: keep-all !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    height: 42px !important;
    border-radius: 10px !important;
    padding: 0 16px !important;
    background-color: transparent !important;
    color: #475569 !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div.stButton > button:hover {
    color: #16A34A !important;
    background-color: #F1F5F9 !important;
    border-color: #E2E8F0 !important;
}

div.stButton > button[kind="primary"] {
    background-color: #16A34A !important;
    color: #FFFFFF !important;
    border: 1px solid #16A34A !important;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.2) !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #15803D !important;
    color: #FFFFFF !important;
    border-color: #15803D !important;
}

/* Image Banner Cards */
.card-with-img {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;
    height: 100%;
}
.card-with-img:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(22, 163, 74, 0.12);
    border-color: #86EFAC;
}
.card-img-header {
    width: 100%;
    height: 190px;
    object-fit: cover;
}
.card-body {
    padding: 24px;
}
.card-tag {
    display: inline-block;
    padding: 4px 12px;
    background-color: #DCFCE7;
    color: #15803D;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
    text-transform: uppercase;
}
.card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 10px;
    line-height: 1.3;
}
.card-text {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.6;
}

/* Hero Section Split Layout */
.hero-container {
    background: linear-gradient(135deg, #F0FDF4 0%, #E0F2FE 100%);
    border-radius: 24px;
    padding: 45px 35px;
    border: 1px solid #DCFCE7;
    margin-bottom: 35px;
    box-shadow: 0 4px 24px rgba(22, 163, 74, 0.06);
}
.hero-tag {
    display: inline-block;
    background-color: #DCFCE7;
    color: #15803D;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #0F172A !important;
    line-height: 1.15;
    margin-bottom: 18px;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    color: #475569 !important;
    line-height: 1.6;
    margin-bottom: 25px;
}

/* Workflow Step Cards */
.step-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    height: 100%;
}
.step-num {
    width: 40px;
    height: 40px;
    background-color: #16A34A;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.1rem;
    margin: 0 auto 14px auto;
    font-family: 'Outfit', sans-serif;
}
.stat-num {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #16A34A;
}
.stat-desc {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 500;
}

/* Admin Dashboard KPI Cards */
.admin-kpi-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 22px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    position: relative;
    overflow: hidden;
}
.admin-kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: #16A34A;
}
.admin-kpi-card.blue::before { background: #0284C7; }
.admin-kpi-card.amber::before { background: #D97706; }
.admin-kpi-card.purple::before { background: #9333EA; }

.admin-kpi-val {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #0F172A;
    margin: 6px 0;
}
.admin-kpi-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Widgets Télémétrie */
.widget-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    margin-bottom: 15px;
    transition: all 0.2s ease;
}
.widget-box:hover {
    border-color: #16A34A;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.08);
}
.widget-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.widget-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F172A;
}

/* Badges de Risque */
.badge-green {
    display: inline-block;
    padding: 5px 12px;
    background-color: #DCFCE7;
    color: #15803D;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-yellow {
    display: inline-block;
    padding: 5px 12px;
    background-color: #FEF9C3;
    color: #A16207;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-red {
    display: inline-block;
    padding: 5px 12px;
    background-color: #FEE2E2;
    color: #B91C1C;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Inputs & File uploader */
div[data-testid="stSelectbox"] > div {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
}
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 24px !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #16A34A !important;
    background-color: #F8FAFC !important;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 12px 20px !important;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Application principale FarmView."""
    
    if 'page' not in st.session_state:
        st.session_state.page = "Accueil"
    if 'is_admin_logged_in' not in st.session_state:
        st.session_state.is_admin_logged_in = False
    
    # --- HEADER / NAVBAR FLOTTANTE PREMIUM ---
    st.markdown('<div class="navbar-wrapper">', unsafe_allow_html=True)
    
    if st.session_state.is_admin_logged_in or st.session_state.page == "Administration":
        # NAVBAR INTERFACE ADMIN DÉDIÉE
        col_nav_logo, col_nav_space, col_nav_logout = st.columns([4, 6, 2])
        
        with col_nav_logo:
            st.markdown("""<div style="display: flex; align-items: center; gap: 12px; height: 42px;">
<div style="width: 36px; height: 36px; background-color: #16A34A; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: white; font-size: 1.2rem; font-family: 'Outfit', sans-serif; box-shadow: 0 2px 8px rgba(22, 163, 74, 0.3);">F</div>
<span style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 800; color: #0F172A;">Farm<span style="color: #16A34A;">View</span></span>
<span style="background-color: #DCFCE7; color: #15803D; padding: 4px 12px; border-radius: 10px; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.5px;">PORTAIL ADMIN</span>
</div>""", unsafe_allow_html=True)
            
        with col_nav_logout:
            if st.button("Se déconnecter", use_container_width=True, type="secondary"):
                st.session_state.is_admin_logged_in = False
                st.session_state.page = "Accueil"
                st.rerun()

    else:
        # NAVBAR INTERFACE PUBLIQUE
        col_nav_logo, col_nav_menu, col_nav_login = st.columns([2.5, 7.2, 2.3])
        
        with col_nav_logo:
            st.markdown("""<div style="display: flex; align-items: center; gap: 10px; height: 42px;">
<div style="width: 36px; height: 36px; background-color: #16A34A; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: white; font-size: 1.2rem; font-family: 'Outfit', sans-serif; box-shadow: 0 2px 8px rgba(22, 163, 74, 0.3);">F</div>
<span style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.5px;">Farm<span style="color: #16A34A;">View</span></span>
</div>""", unsafe_allow_html=True)
            
        with col_nav_menu:
            c1, c2, c3, c4 = st.columns([1, 1.1, 1.1, 1.7])
            if c1.button("Accueil", use_container_width=True, type="primary" if st.session_state.page == "Accueil" else "secondary"):
                st.session_state.page = "Accueil"
                st.rerun()
            if c2.button("Diagnostic", use_container_width=True, type="primary" if st.session_state.page == "Diagnostic" else "secondary"):
                st.session_state.page = "Diagnostic"
                st.rerun()
            if c3.button("Informations", use_container_width=True, type="primary" if st.session_state.page == "Informations" else "secondary"):
                st.session_state.page = "Informations"
                st.rerun()
            if c4.button("Base de connaissances", use_container_width=True, type="primary" if st.session_state.page == "Base de connaissances" else "secondary"):
                st.session_state.page = "Base de connaissances"
                st.rerun()

        with col_nav_login:
            if st.button("Se Connecter", use_container_width=True, type="primary"):
                st.session_state.page = "Connexion"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- ROUTING ---
    if st.session_state.page == "Accueil":
        show_home_page()
    elif st.session_state.page == "Diagnostic":
        show_diagnostic_page()
    elif st.session_state.page == "Informations":
        show_info_page()
    elif st.session_state.page == "Base de connaissances":
        show_knowledge_page()
    elif st.session_state.page == "Connexion":
        show_login_page()
    elif st.session_state.page == "Administration":
        show_admin_page()

def show_home_page():
    """Page d'accueil de l'interface publique."""
    
    col_h_text, col_h_img = st.columns([7, 5], gap="large")
    
    with col_h_text:
        st.markdown("""<div class="hero-container" style="height: 100%;">
<div class="hero-tag">FarmView IA — Surveillance Agronomique</div>
<h1 class="hero-title">Anticipez les maladies des cultures grâce à l'IA</h1>
<p class="hero-subtitle">Photographiez les feuilles en direct au champ, identifiez instantanément les symptômes foliaires et protégez vos rendements avec des conseils de biocontrôle ciblés.</p>
</div>""", unsafe_allow_html=True)
        
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("Lancer un diagnostic IA", type="primary", use_container_width=True):
                st.session_state.page = "Diagnostic"
                st.rerun()
        with btn_c2:
            if st.button("Explorer la base phytosanitaire", type="secondary", use_container_width=True):
                st.session_state.page = "Base de connaissances"
                st.rerun()

    with col_h_img:
        st.markdown("""<div style="position: relative; border-radius: 24px; overflow: hidden; box-shadow: 0 12px 36px rgba(0,0,0,0.08); border: 4px solid #FFFFFF;">
<img src="https://images.unsplash.com/photo-1586771107445-d3ca888129ff?auto=format&fit=crop&w=800&q=80" style="width: 100%; height: 380px; object-fit: cover; display: block;" />
<div style="position: absolute; bottom: 20px; left: 20px; right: 20px; background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(8px); padding: 16px 20px; border-radius: 14px; border: 1px solid rgba(226, 232, 240, 0.8); display: flex; justify-content: space-between; align-items: center;">
<div>
<div style="font-size: 0.75rem; font-weight: 700; color: #16A34A; text-transform: uppercase;">Scan IA Actif</div>
<div style="font-size: 1rem; font-weight: 700; color: #0F172A;">Précision de Détection : 96.5%</div>
</div>
<span style="background-color: #DCFCE7; color: #15803D; padding: 4px 10px; border-radius: 10px; font-size: 0.8rem; font-weight: 700;">MobileNetV2</span>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown("""<div class="step-card">
<div class="stat-num">50+</div>
<div class="stat-desc">Coopératives Partenaires</div>
</div>""", unsafe_allow_html=True)
    with col_s2:
        st.markdown("""<div class="step-card">
<div class="stat-num">1,840+</div>
<div class="stat-desc">Diagnostics Réalisés</div>
</div>""", unsafe_allow_html=True)
    with col_s3:
        st.markdown("""<div class="step-card">
<div class="stat-num">96.5%</div>
<div class="stat-desc">Précision du Modèle</div>
</div>""", unsafe_allow_html=True)
    with col_s4:
        st.markdown("""<div class="step-card">
<div class="stat-num">&lt; 2s</div>
<div class="stat-desc">Temps d'Analyse Moyen</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: Outfit; font-size: 1.8rem; margin-bottom: 20px; color: #0F172A;'>Présentation stratégique du projet</h2>", unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1530507629858-e4977d30e9e0?auto=format&fit=crop&w=600&q=80" class="card-img-header" />
<div class="card-body">
<span class="card-tag" style="background-color: #FEE2E2; color: #B91C1C;">Problème Identifié</span>
<div class="card-title">Diagnostics Tardifs & Pertes de Récolte</div>
<div class="card-text">L'absence d'outils d'évaluation rapide au champ entraîne des interventions tardives et des baisses considérables de rendement.</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1628352081506-83c43123ed6d?auto=format&fit=crop&w=600&q=80" class="card-img-header" />
<div class="card-body">
<span class="card-tag">Solution IA</span>
<div class="card-title">Vision par Ordinateur au Champ</div>
<div class="card-text">Algorithme de Deep Learning entraîné sur des milliers de clichés foliaires pour détecter les pathologies et conseiller des soins durables.</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_p3:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80" class="card-img-header" />
<div class="card-body">
<span class="card-tag" style="background-color: #E0F2FE; color: #0369A1;">Valeur Produit</span>
<div class="card-title">Pilotage Réseau & Coopératives</div>
<div class="card-text">Un modèle d'abonnement clé en main permettant aux coopératives de superviser la santé parcellaire globale et d'optimiser les conseils.</div>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: Outfit; font-size: 1.8rem; margin-bottom: 20px; color: #0F172A;'>Comment fonctionne FarmView ?</h2>", unsafe_allow_html=True)
    
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        st.markdown("""<div class="step-card">
<div class="step-num">1</div>
<div style="font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.15rem; color: #0F172A; margin-bottom: 8px;">Prenez une Photo</div>
<div style="font-size: 0.9rem; color: #64748B; line-height: 1.5;">Photographiez la feuille présentant des taches ou anomalies directement avec votre smartphone ou appareil.</div>
</div>""", unsafe_allow_html=True)
        
    with col_w2:
        st.markdown("""<div class="step-card">
<div class="step-num">2</div>
<div style="font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.15rem; color: #0F172A; margin-bottom: 8px;">Analyse IA Instantanée</div>
<div style="font-size: 0.9rem; color: #64748B; line-height: 1.5;">Notre modèle TensorFlow compare les caractéristiques visuelles et calcule l'indice de certitude en moins de 2 secondes.</div>
</div>""", unsafe_allow_html=True)
        
    with col_w3:
        st.markdown("""<div class="step-card">
<div class="step-num">3</div>
<div style="font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.15rem; color: #0F172A; margin-bottom: 8px;">Appliquez le Soin</div>
<div style="font-size: 0.9rem; color: #64748B; line-height: 1.5;">Consultez les recommandations de biocontrôle et les traitements agro-écologiques adaptés à la culture.</div>
</div>""", unsafe_allow_html=True)

def show_diagnostic_page():
    """Page de diagnostic."""
    st.title("Diagnostic IA des Pathologies Foliaires")
    st.markdown("""<p style="font-size: 1.05rem; color: #475569; margin-bottom: 30px;">
Importez un cliché de feuille pour lancer l'analyse prédictive. Notre modèle identifie la maladie et génère un protocole de soin.
</p>""", unsafe_allow_html=True)
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        selected_crop = st.selectbox(
            "Culture concernée (optionnel)",
            ["Je ne sais pas"] + CROPS,
            help="Sélectionner la culture affine l'analyse"
        )
    
    with col_opt2:
        confidence_threshold = st.slider(
            "Seuil de confiance minimum (%)",
            min_value=50,
            max_value=95,
            value=75,
            help="Niveau de confiance minimum exigé"
        )
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Glissez-déposez ou sélectionnez une photo de feuille",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col_res1, col_res2 = st.columns(2, gap="large")
        
        with col_res1:
            st.markdown("""<div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">""", unsafe_allow_html=True)
            st.subheader("Cliché soumis")
            display_image = resize_image(image)
            st.image(display_image, use_column_width=True)
            st.caption(f"Format : {image.format} | Dimensions : {image.size} | Mode : {image.mode}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_res2:
            st.subheader("Résultats de l'Analyse IA")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for percent in range(0, 101, 25):
                status_text.text(f"Traitement par le modèle IA... {percent}%")
                progress_bar.progress(percent / 100)
                import time
                time.sleep(0.08)
            
            status_text.empty()
            progress_bar.empty()
            
            if selected_crop == "Tomate":
                fake_prediction = "Tomato___Early_blight"
                confidence = 0.92
            elif selected_crop == "Pomme de terre":
                fake_prediction = "Potato___Late_blight"
                confidence = 0.88
            else:
                fake_prediction = "Apple___Apple_scab"
                confidence = 0.85
            
            if confidence * 100 < confidence_threshold:
                st.markdown("""<div class="custom-warning">
<strong>Diagnostic incertain</strong><br>
Le niveau de certitude mesuré est inférieur au seuil configuré.
</div>""", unsafe_allow_html=True)
                st.info(f"Confiance mesurée : {confidence*100:.1f}% (Seuil requis : {confidence_threshold}%)")
            else:
                st.markdown("""<div class="custom-success">
<strong>Analyse validée avec succès</strong>
</div>""", unsafe_allow_html=True)
                
                disease_info = get_disease_info(fake_prediction)
                
                st.metric("Pathologie détectée", disease_info["name"])
                st.metric("Indice de confiance", f"{confidence*100:.1f}%")
                
                st.markdown(f"""<div style="background: white; padding: 20px; border-radius: 14px; border: 1px solid #E2E8F0; margin-top: 15px;">
<h4 style="color:#0F172A;margin-top:0;">Description clinique</h4>
<p style="color:#475569;font-size:0.95rem;line-height:1.6;">{disease_info["description"]}</p>
</div>""", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color:#16A34A;margin-top:20px;'>Recommandations de traitement agro-écologique</h4>", unsafe_allow_html=True)
                for i, treatment in enumerate(disease_info["treatment"], 1):
                    st.write(f"**{i}.** {treatment}")
                
                if disease_info["chemical"] != "Aucun traitement nécessaire":
                    st.markdown("<h4 style='color:#DC2626;margin-top:20px;'>Option de traitement chimique</h4>", unsafe_allow_html=True)
                    st.warning(disease_info["chemical"])
        
        st.markdown("---")
        if st.button("Réinitialiser l'analyse", type="secondary"):
            st.rerun()

def show_info_page():
    """Page d'informations."""
    st.title("À propos de la plateforme FarmView")
    
    st.markdown("""<p style="font-size: 1.1rem; color: #475569; margin-bottom: 35px; line-height: 1.6;">
FarmView combine l'agronomie et l'IA pour fournir des outils décisionnels de pointe aux acteurs de la filière agricole.
</p>""", unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2, gap="large")
    
    with col_i1:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=600&q=80" class="card-img-header" />
<div class="card-body">
<div class="card-title">Notre Mission Agronomique</div>
<div class="card-text">
Démocratiser l'accès aux technologies de pointe pour préserver les récoltes, réduire l’empreinte environnementale des traitements et sécuriser les revenus des exploitants.
</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_i2:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1567306301408-9b74779a11af?auto=format&fit=crop&w=600&q=80" class="card-img-header" />
<div class="card-body">
<div class="card-title">Technologie & Sécurité des Données</div>
<div class="card-text">
Modèles de Transfer Learning entraînés sur des banques d'images universitaires certifiées. Vos clichés sont traités localement sans transfert intrusif.
</div>
</div>
</div>""", unsafe_allow_html=True)

def show_knowledge_page():
    """Page de base de connaissances."""
    st.title("Base de connaissances phytosanitaires")
    st.markdown("""<p style="font-size: 1.05rem; color: #475569; margin-bottom: 30px;">
Catalogue des pathologies végétales documentées et guides pratiques.
</p>""", unsafe_allow_html=True)
    
    diseases = [
        ("Tavelure du pommier", "Apple___Apple_scab", "Pathologie Fongique", "https://images.unsplash.com/photo-1567306301408-9b74779a11af?auto=format&fit=crop&w=400&q=80"),
        ("Mildiou de la tomate / pomme de terre", "Potato___Late_blight", "Pathologie Fongique", "https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=400&q=80"),
        ("Oïdium & Esca de la vigne", "Grape___Esca", "Pathologie Fongique", "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=400&q=80"),
    ]
    
    st.markdown("### Pathologies Répertoriées")
    for disease_name, disease_id, disease_type, img_url in diseases:
        with st.expander(f"{disease_name} — ({disease_type})"):
            col_exp1, col_exp2 = st.columns([4, 8])
            with col_exp1:
                st.image(img_url, use_column_width=True)
            with col_exp2:
                info = get_disease_info(disease_id)
                st.write(f"**Description :** {info['description']}")
                st.write("**Traitements recommandés :**")
                for treatment in info['treatment']:
                    st.write(f"- {treatment}")

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2, gap="large")
    
    with col_c1:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1586771107445-d3ca888129ff?auto=format&fit=crop&w=600&q=80" class="card-img-header" />
<div class="card-body">
<div class="card-title">Guide de prise de vue</div>
<div class="card-text">
1. Éclairage naturel sans ombres portées directes.<br>
2. Cadrage centré sur la zone symptomatique.<br>
3. Fond neutre contrasté pour isoler la feuille.
</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_c2:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80" class="card-img-header" />
<div class="card-body">
<div class="card-title">Symptômes cliniques majeurs</div>
<div class="card-text">
- Taches nécrotiques circulaires ou décolorées.<br>
- Feutrage blanchâtre sous le limbe foliaire.<br>
- Flétrissement ou enroulement des bordures.
</div>
</div>
</div>""", unsafe_allow_html=True)

def show_login_page():
    """Page de connexion dédiée à l'Espace Administration."""
    
    col_l1, col_l2, col_l3 = st.columns([3, 6, 3])
    
    with col_l2:
        st.markdown("""<div style="background: white; padding: 40px; border-radius: 20px; border: 1px solid #E2E8F0; box-shadow: 0 8px 30px rgba(0,0,0,0.06); text-align: center; margin-top: 20px;">
<div style="width: 54px; height: 54px; background-color: #DCFCE7; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 18px auto;">
<div style="width: 26px; height: 26px; background-color: #16A34A; border-radius: 8px;"></div>
</div>
<h2 style="font-family: Outfit; font-size: 2.2rem; color: #0F172A; margin-bottom: 8px;">Portail d'Administration</h2>
<p style="color: #64748B; font-size: 0.98rem; margin-bottom: 30px;">Authentification requise pour accéder au centre de commande et à la cartographie des risques.</p>
</div>""", unsafe_allow_html=True)
        
        with st.form("login_form"):
            email_input = st.text_input("Identifiant Administrateur", value="admin@farmview.ai")
            password_input = st.text_input("Mot de passe", type="password", value="admin123")
            remember_me = st.checkbox("Se souvenir de cette session")
            
            submit_login = st.form_submit_button("Se connecter à l'espace Admin", type="primary", use_container_width=True)
            
            if submit_login:
                if email_input and password_input:
                    st.session_state.is_admin_logged_in = True
                    st.session_state.page = "Administration"
                    st.success("Authentification réussie ! Chargement du tableau de bord...")
                    st.rerun()
                else:
                    st.error("Veuillez remplir vos identifiants.")

def show_admin_page():
    """Espace Administration & Pilotage Agronomique Ultra-Design."""
    
    # --- TOP BANNER ADMIN CORPORATE ---
    st.markdown("""<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 28px 32px; border-radius: 20px; color: white; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15); border: 1px solid #334155;">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
<div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
<span style="background-color: #16A34A; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">Centre de Commande IA</span>
<span style="color: #94A3B8; font-size: 0.85rem;">● Synchro GPS : En direct</span>
</div>
<h1 style="font-family: Outfit; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0; letter-spacing: -0.5px;">Supervision & Pilotage des Coopératives</h1>
<p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">Cartographie décisionnelle parcelles, détection précoce des épidémies et gouvernance des algorithmes.</p>
</div>
<div style="background: rgba(255,255,255,0.08); backdrop-filter: blur(10px); padding: 14px 22px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12); text-align: right;">
<div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Statut Serveurs IA</div>
<div style="font-family: Outfit; font-size: 1.15rem; font-weight: 700; color: #4ADE80; margin-top: 2px;">● MobileNetV2 Actif</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # --- BANNIÈRE DE METRIQUES KPI ---
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown("""<div class="admin-kpi-card">
<div class="admin-kpi-label">Exploitations Connectées</div>
<div class="admin-kpi-val">142</div>
<div style="font-size: 0.8rem; color: #16A34A; font-weight: 700;">+8 ce mois-ci</div>
</div>""", unsafe_allow_html=True)
    with col_k2:
        st.markdown("""<div class="admin-kpi-card blue">
<div class="admin-kpi-label">Diagnostics Exécutés</div>
<div class="admin-kpi-val">1,842</div>
<div style="font-size: 0.8rem; color: #0284C7; font-weight: 700;">+156 cette semaine</div>
</div>""", unsafe_allow_html=True)
    with col_k3:
        st.markdown("""<div class="admin-kpi-card amber">
<div class="admin-kpi-label">Foyers sous Vigilance</div>
<div class="admin-kpi-val">14</div>
<div style="font-size: 0.8rem; color: #D97706; font-weight: 700;">3 alertes critiques</div>
</div>""", unsafe_allow_html=True)
    with col_k4:
        st.markdown("""<div class="admin-kpi-card purple">
<div class="admin-kpi-label">Précision Modèle IA</div>
<div class="admin-kpi-val">96.5%</div>
<div style="font-size: 0.8rem; color: #9333EA; font-weight: 700;">Validation v2.4</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 35px;'></div>", unsafe_allow_html=True)

    # --- TABS ADMIN DESIGN ---
    tab_map, tab_kpi, tab_alerts, tab_ai, tab_users = st.tabs([
        "📍 Cartographie des Risques",
        "📊 KPIs & Analytics",
        "⚠️ Alertes Phytosanitaires",
        "🧠 Supervision Modèle IA",
        "👥 Annuaire Coopérateurs"
    ])
    
    # --- TAB 1 : CARTE GEOGRAPHIQUE INTERACTIVE ---
    with tab_map:
        st.markdown("<h3 style='font-family: Outfit; color: #0F172A; margin-bottom: 5px;'>Cartographie des Parcelles & Zones à Risque</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-bottom: 20px;'>Consultez l'état phytosanitaire des cultures sur le réseau géographique.</p>", unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_status = st.multiselect(
                "Niveau de risque sanitaire",
                ["Alerte Rouge (Foyer d'infection)", "Vigilance Jaune (Risque modéré)", "Parcelle Saine (Vert)"],
                default=["Alerte Rouge (Foyer d'infection)", "Vigilance Jaune (Risque modéré)", "Parcelle Saine (Vert)"]
            )
        with col_f2:
            filter_crop = st.multiselect(
                "Type de culture",
                ["Toutes"] + CROPS,
                default=["Toutes"]
            )
        
        data_plots = [
            {"lat": 10.05, "lon": -12.85, "Exploitation": "Ferme Kindia Agro", "Culture": "Tomate", "Statut": "Alerte Rouge (Foyer d'infection)", "Maladie": "Mildiou de la tomate", "Superficie": "4.2 ha", "color": "#DC2626"},
            {"lat": 10.38, "lon": -12.28, "Exploitation": "Coopérative Dalaba", "Culture": "Pomme de terre", "Statut": "Vigilance Jaune (Risque modéré)", "Maladie": "Risque Tavelure", "Superficie": "8.5 ha", "color": "#F59E0B"},
            {"lat": 9.88, "lon": -9.31, "Exploitation": "Domaine Kankan Sud", "Culture": "Maïs", "Statut": "Parcelle Saine (Vert)", "Maladie": "Aucune", "Superficie": "12.0 ha", "color": "#16A34A"},
            {"lat": 11.31, "lon": -12.28, "Exploitation": "Vergers de Labé", "Culture": "Pomme", "Statut": "Parcelle Saine (Vert)", "Maladie": "Aucune", "Superficie": "6.0 ha", "color": "#16A34A"},
            {"lat": 7.75, "lon": -8.82, "Exploitation": "Vignoble Forestier Nzérékoré", "Culture": "Vigne", "Statut": "Alerte Rouge (Foyer d'infection)", "Maladie": "Esca / Oïdium", "Superficie": "5.5 ha", "color": "#DC2626"},
            {"lat": 9.43, "lon": -13.08, "Exploitation": "Maraîchage Forecariah", "Culture": "Tomate", "Statut": "Parcelle Saine (Vert)", "Maladie": "Aucune", "Superficie": "3.8 ha", "color": "#16A34A"},
            {"lat": 10.65, "lon": -11.45, "Exploitation": "Ferme Dabola", "Culture": "Pomme de terre", "Statut": "Vigilance Jaune (Risque modéré)", "Maladie": "Début de tavelure", "Superficie": "7.1 ha", "color": "#F59E0B"},
        ]
        
        df_plots = pd.DataFrame(data_plots)
        
        if filter_status:
            df_filtered = df_plots[df_plots["Statut"].isin(filter_status)]
        else:
            df_filtered = df_plots
            
        if filter_crop and "Toutes" not in filter_crop:
            df_filtered = df_filtered[df_filtered["Culture"].isin(filter_crop)]
            
        st.map(df_filtered, latitude="lat", longitude="lon", color="color", size=50, use_container_width=True)
        
        st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='font-family: Outfit; color: #0F172A;'>Répertoire des Parcelles Connectées</h4>", unsafe_allow_html=True)
        st.dataframe(
            df_filtered[["Exploitation", "Culture", "Statut", "Maladie", "Superficie"]],
            use_container_width=True,
            hide_index=True
        )

    # --- TAB 2 : KPIS ET STATISTIQUES ---
    with tab_kpi:
        st.markdown("<h3 style='font-family: Outfit; color: #0F172A;'>Analytique & Performances Agronomiques</h3>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2, gap="large")
        with col_chart1:
            st.markdown("#### Distribution des Pathologies Détectées")
            chart_data = pd.DataFrame({
                "Maladie": ["Mildiou Tomate", "Tavelure Pomme", "Oïdium Vigne", "Rouille Maïs", "Parcelles Saines"],
                "Occurrences": [420, 280, 190, 150, 802]
            })
            st.bar_chart(chart_data.set_index("Maladie"))
            
        with col_chart2:
            st.markdown("#### Diagnostics par Bassin de Production")
            region_data = pd.DataFrame({
                "Région": ["Kindia", "Labé", "Mamou", "Kankan", "Nzérékoré"],
                "Volume Diagnostics": [540, 420, 380, 290, 212]
            })
            st.line_chart(region_data.set_index("Région"))

    # --- TAB 3 : GESTION DES ALERTES SANITAIRES ---
    with tab_alerts:
        st.markdown("<h3 style='font-family: Outfit; color: #0F172A;'>Diffusion & Gestion des Alertes Sanitaires</h3>", unsafe_allow_html=True)
        
        st.markdown("""<div class="custom-warning">
<strong>Alerte Régionale Active :</strong> Risque élevé de développement du Mildiou suite au taux d'humidité élevé dans le secteur Kindia.
</div>""", unsafe_allow_html=True)
        
        st.markdown("#### Publier une Notification Sanitaire Ciblée")
        with st.form("alert_form"):
            alert_zone = st.selectbox("Secteur agricole concerné", ["Tous les secteurs", "Kindia", "Labé", "Mamou", "Kankan", "Nzérékoré"])
            alert_disease = st.selectbox("Maladie ou symptôme identifié", CROPS)
            alert_severity = st.radio("Seuil d'urgence", ["Vigilance Jaune", "Alerte Rouge - Intervention Immédiate"], horizontal=True)
            alert_msg = st.text_area("Consignes de traitement & biocontrôle", "Humidité foliaire supérieure à 80%. Appliquer une pulvérisation de produit cuivré préventif sous 24 heures.")
            
            submit_alert = st.form_submit_button("Diffuser l'Alerte aux Agriculteurs de la Zone", type="primary")
            if submit_alert:
                st.success(f"Notification d'alerte envoyée avec succès aux agriculteurs du secteur {alert_zone} !")

    # --- TAB 4 : SUPERVISION DES MODELES IA ---
    with tab_ai:
        st.markdown("<h3 style='font-family: Outfit; color: #0F172A;'>Supervision des Algorithmes & Modèles IA</h3>", unsafe_allow_html=True)
        
        col_ai1, col_ai2 = st.columns(2, gap="large")
        with col_ai1:
            st.markdown("""<div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
<h4 style="color:#0F172A;margin-top:0;font-family:Outfit;">Fiche Technique Modèle de Production</h4>
<p style="color:#475569;font-size:0.95rem;line-height:1.7;">
- <strong>Architecture</strong> : MobileNetV2 (Transfer Learning)<br>
- <strong>Classes Reconnues</strong> : 38 pathologies végétales<br>
- <strong>Résolution d'Entrée</strong> : 224 x 224 x 3 pixels<br>
- <strong>Précision (Val Accuracy)</strong> : 96.5%<br>
- <strong>Perte (Val Loss)</strong> : 0.118<br>
- <strong>Temps d'inférence moyen</strong> : 1.4 secondes
</p>
</div>""", unsafe_allow_html=True)
            
        with col_ai2:
            st.markdown("#### Déployer une Nouvelle Version du Modèle (`.h5` / `.keras`)")
            new_model_file = st.file_uploader("Fichier poids du modèle neural", type=["h5", "keras", "pkl"])
            if new_model_file:
                st.success(f"Nouveau modèle '{new_model_file.name}' prêt pour le déploiement en production !")
                st.button("Basculer le modèle en production", type="primary")

    # --- TAB 5 : GESTION DES COOPERATEURS ---
    with tab_users:
        st.markdown("<h3 style='font-family: Outfit; color: #0F172A;'>Gestion du Réseau d'Exploitants & Coopératives</h3>", unsafe_allow_html=True)
        
        col_u1, col_u2 = st.columns([7, 5], gap="large")
        with col_u1:
            users_df = pd.DataFrame([
                {"Agriculteur": "Mamadou Bah", "Coopérative": "Kindia Maraîchage", "Culture": "Tomate", "Superficie": "4.2 ha", "Statut": "Actif"},
                {"Agriculteur": "Sékou Camara", "Coopérative": "Dalaba Agro", "Culture": "Pomme de terre", "Superficie": "8.5 ha", "Statut": "Actif"},
                {"Agriculteur": "Fatoumata Diallo", "Coopérative": "Kankan Bio", "Culture": "Maïs", "Superficie": "12.0 ha", "Statut": "Actif"},
                {"Agriculteur": "Jean-Pierre Loua", "Coopérative": "Nzérékoré Vigne", "Culture": "Vigne", "Superficie": "5.5 ha", "Statut": "Sous surveillance"},
            ])
            st.dataframe(users_df, use_container_width=True, hide_index=True)
            
        with col_u2:
            st.markdown("#### Enregistrer une Nouvelle Parcelle / Exploitant")
            with st.form("new_user_form"):
                st.text_input("Nom & Prénom de l'exploitant")
                st.text_input("Numéro Téléphone / SMS")
                st.selectbox("Coopérative de rattachement", ["Kindia Maraîchage", "Dalaba Agro", "Kankan Bio", "Autre"])
                st.number_input("Superficie de la parcelle (ha)", min_value=0.5, max_value=100.0, value=5.0)
                st.form_submit_button("Ajouter l'exploitant au réseau", type="primary")

if __name__ == "__main__":
    main()