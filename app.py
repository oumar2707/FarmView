"""
FarmView - Application de diagnostic agricole par IA
Interface ultra-design avec Espace Admin Premium, Carte Géographique & Design Corporate
"""

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
import sys
import os

# Ajouter le dossier src au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer nos modules
from src.data.disease_info import get_disease_info, CROPS
from src.models.predictor import predictor
from src.utils.admin_auth import authenticate_admin
from streamlit.errors import StreamlitSecretNotFoundError

# Configuration de la page
st.set_page_config(
    page_title="FarmView - Diagnostic Agricole IA",
    page_icon="🌱",
    layout="wide"
)

from src.ui.components import inject_theme, load_test_metrics, render_status_badge

inject_theme()


def _admin_credentials():
    email = os.environ.get("FARMVIEW_ADMIN_EMAIL")
    password = os.environ.get("FARMVIEW_ADMIN_PASSWORD")
    try:
        admin_secrets = st.secrets.get("admin", {})
    except StreamlitSecretNotFoundError:
        admin_secrets = {}
    if not isinstance(admin_secrets, dict):
        admin_secrets = {}
    return (
        email or admin_secrets.get("email"),
        password or admin_secrets.get("password"),
    )


def main():
    """Application principale FarmView."""
    
    if 'page' not in st.session_state:
        st.session_state.page = "Accueil"
    if 'is_admin_logged_in' not in st.session_state:
        st.session_state.is_admin_logged_in = False
    if st.session_state.page == "Administration" and not st.session_state.is_admin_logged_in:
        st.session_state.page = "Connexion"
    
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
    model_state = (
        "Un modèle entraîné est chargé."
        if predictor.model_loaded
        else "Aucun modèle entraîné valide n'est disponible : aucun diagnostic ne sera produit."
    )
    
    with col_h_text:
        st.markdown("""<div class="hero-container" style="height: 100%;">
<div class="hero-tag">FarmView — aide à la décision</div>
<h1 class="hero-title">Mieux observer la santé de vos cultures</h1>
<p class="hero-subtitle">Importez une photo pour vérifier si un modèle entraîné est disponible et consulter les informations phytosanitaires adaptées.</p>
</div>""", unsafe_allow_html=True)
        st.info(model_state)
        
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
        st.markdown('<div class="hero-container"><span class="hero-tag">Parcours de diagnostic</span><h3>Photo → analyse vérifiée → ressources</h3><p class="hero-subtitle">La démonstration n’invente pas de résultat si le modèle est absent ou invalide.</p></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown("""<div class="step-card">
<div class="card-tag">1</div>
<h3>Importer</h3>
<div class="card-text">Fournissez une photo JPG, JPEG ou PNG de la feuille observée.</div>
</div>""", unsafe_allow_html=True)
    with col_s2:
        st.markdown("""<div class="step-card">
<div class="card-tag">2</div>
<h3>Vérifier</h3>
<div class="card-text">L’application indique explicitement si l’inférence a été faite par un modèle entraîné.</div>
</div>""", unsafe_allow_html=True)
    with col_s3:
        st.markdown("""<div class="step-card">
<div class="card-tag">3</div>
<h3>Consulter</h3>
<div class="card-text">Utilisez les informations de maladie comme aide, puis demandez conseil à un spécialiste.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: Outfit; font-size: 1.8rem; margin-bottom: 20px; color: #0F172A;'>Présentation stratégique du projet</h2>", unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1530507629858-e4977d30e9e0?auto=format&fit=crop&w=600&q=80" alt="Illustration agricole, non utilisée pour un diagnostic" class="card-img-header" />
<div class="card-body">
<span class="card-tag" style="background-color: #FEE2E2; color: #B91C1C;">Problème Identifié</span>
<div class="card-title">Diagnostics Tardifs & Pertes de Récolte</div>
<div class="card-text">L'absence d'outils d'évaluation rapide au champ entraîne des interventions tardives et des baisses considérables de rendement.</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1628352081506-83c43123ed6d?auto=format&fit=crop&w=600&q=80" alt="Illustration de culture agricole, non utilisée pour un diagnostic" class="card-img-header" />
<div class="card-body">
<span class="card-tag">Solution IA</span>
<div class="card-title">Vision par Ordinateur au Champ</div>
<div class="card-text">Le pipeline permet d’utiliser un modèle entraîné quand il est présent et validé. Aucun résultat de démonstration ne remplace une prédiction.</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_p3:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80" alt="Illustration de paysage agricole, non utilisée pour un diagnostic" class="card-img-header" />
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

    st.warning("FarmView est une aide à la décision et ne remplace pas l’avis d’un agronome ou d’un conseiller phytosanitaire.")

def show_diagnostic_page():
    """Run a real model inference or report why no diagnosis is available."""
    from pathlib import Path

    st.title("Diagnostic des feuilles")
    st.write(
        "Importez une image. Une maladie ne sera affichée que si un modèle entraîné "
        "et compatible produit effectivement une prédiction."
    )
    if not predictor.model_loaded:
        st.warning("Modèle absent ou invalide : le diagnostic est indisponible.")

    selected_crop = st.selectbox(
        "Culture observée (facultatif)", ["Je ne sais pas"] + CROPS
    )
    uploaded_file = st.file_uploader(
        "Importer une photo de feuille", type=["jpg", "jpeg", "png"]
    )

    sample_dir = Path(__file__).resolve().parent / "data" / "demo_samples"
    demo_samples = sorted(
        path for path in sample_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if sample_dir.is_dir() else []
    sample_labels = {path.parent.name: path for path in demo_samples}
    selected_sample = st.selectbox(
        "Image d'exemple (données de démonstration)",
        ["Aucune"] + list(sample_labels),
    )
    use_sample = st.button(
        "Essayer avec une image d'exemple", disabled=selected_sample == "Aucune"
    )

    image_source = uploaded_file
    if use_sample and selected_sample in sample_labels:
        image_source = sample_labels[selected_sample]
    if image_source is None:
        st.info("Ajoutez une image pour commencer. Aucune analyse n'a encore été effectuée.")
        return

    try:
        with Image.open(image_source) as opened_image:
            opened_image.load()
            image = opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Image invalide ou corrompue : {exc}")
        return

    st.image(image, caption="Image soumise, convertie en RGB pour l'inférence", use_container_width=True)
    if predictor.model_loaded:
        with st.spinner("Inférence du modèle entraîné…"):
            result = predictor.predict(image)
    else:
        result = predictor.predict(image)

    status = result.get("diagnostic_status")
    if status == "model_unavailable":
        st.error(result.get("message") or "Aucun modèle valide n'est disponible.")
        return
    if status == "inference_error":
        st.error(result.get("message") or "L'inférence a échoué.")
        return
    if result.get("source") != "trained_model" or status not in {
        "healthy",
        "disease_detected",
        "uncertain",
    }:
        st.error("Aucune prédiction vérifiable d'un modèle entraîné n'est disponible.")
        return

    confidence = result.get("confidence")
    predicted_class = result.get("predicted_class")
    if not isinstance(confidence, (int, float)) or not predicted_class:
        st.error("Le modèle a renvoyé un résultat incomplet : aucun diagnostic n'est affiché.")
        return

    if confidence < 0.75 or status == "uncertain":
        status = "uncertain"
    label = {
        "healthy": "Classe saine prédite par le modèle",
        "disease_detected": "Classe de maladie prédite par le modèle",
        "uncertain": "Résultat incertain",
    }.get(status, "Résultat incertain")
    render_status_badge(status, label)
    st.caption("Source : modèle entraîné — score Softmax, non calibré.")

    disease_info = get_disease_info(predicted_class)
    st.subheader(disease_info["name"])
    st.write(f"Score du modèle : {confidence:.1%}")
    st.progress(min(max(float(confidence), 0.0), 1.0))

    if selected_crop != "Je ne sais pas" and selected_crop != disease_info["crop"]:
        st.warning(
            f"La classe prédite concerne {disease_info['crop']}, alors que "
            f"« {selected_crop} » a été sélectionnée. Vérifiez l'image et le résultat."
        )
    if status == "uncertain":
        st.warning("Résultat incertain : aucune recommandation de traitement n'est proposée.")
    else:
        st.write(disease_info["description"])
        if status == "disease_detected":
            st.markdown("#### Informations de prise en charge")
            for treatment in disease_info["treatment"]:
                st.write(f"- {treatment}")
            if disease_info["chemical"] != "Aucun traitement nécessaire":
                st.warning(disease_info["chemical"])
        else:
            st.success("Aucune maladie n'est prédite pour cette image par le modèle.")

    top_predictions = result.get("top_predictions", [])
    if top_predictions:
        st.markdown("#### Classes les plus probables")
        for class_name, score in top_predictions[:3]:
            st.write(f"{get_disease_info(class_name)['name']} — {score:.1%}")
            st.progress(min(max(float(score), 0.0), 1.0))

    st.caption("Aide à la décision uniquement ; faites confirmer le diagnostic par un agronome.")

def show_info_page():
    """Page d'informations."""
    st.title("À propos de la plateforme FarmView")
    
    st.markdown("""<p style="font-size: 1.1rem; color: #475569; margin-bottom: 35px; line-height: 1.6;">
FarmView combine l'agronomie et l'IA pour fournir des outils décisionnels de pointe aux acteurs de la filière agricole.
</p>""", unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2, gap="large")
    
    with col_i1:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=600&q=80" alt="Illustration agricole générique" class="card-img-header" />
<div class="card-body">
<div class="card-title">Notre Mission Agronomique</div>
<div class="card-text">
Démocratiser l'accès aux technologies de pointe pour préserver les récoltes, réduire l’empreinte environnementale des traitements et sécuriser les revenus des exploitants.
</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_i2:
        st.markdown("""<div class="card-with-img">
<img src="https://images.unsplash.com/photo-1567306301408-9b74779a11af?auto=format&fit=crop&w=600&q=80" alt="Illustration agricole générique" class="card-img-header" />
<div class="card-body">
<div class="card-title">Technologie & Sécurité des Données</div>
<div class="card-text">
Le pipeline d'inférence est prévu pour un modèle MobileNetV2. Aucun modèle utilisable n'est fourni actuellement ; aucun diagnostic ne sera simulé.
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
                st.image(
                    img_url,
                    caption="Illustration générique, non représentative d'un diagnostic.",
                    use_container_width=True,
                )
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
<img src="https://images.unsplash.com/photo-1586771107445-d3ca888129ff?auto=format&fit=crop&w=600&q=80" alt="Illustration agricole générique" class="card-img-header" />
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
<img src="https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80" alt="Illustration agricole générique" class="card-img-header" />
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
            email_input = st.text_input("Identifiant Administrateur")
            password_input = st.text_input("Mot de passe", type="password")
            
            submit_login = st.form_submit_button("Se connecter à l'espace Admin", type="primary", use_container_width=True)
            
            if submit_login:
                expected_email, expected_password = _admin_credentials()
                if authenticate_admin(
                    email_input,
                    password_input,
                    expected_email,
                    expected_password,
                ):
                    st.session_state.is_admin_logged_in = True
                    st.session_state.page = "Administration"
                    st.success("Authentification réussie ! Chargement du tableau de bord...")
                    st.rerun()
                else:
                    st.error(
                        "Identifiants invalides ou configuration administrateur absente. "
                        "Configurez FARMVIEW_ADMIN_EMAIL et FARMVIEW_ADMIN_PASSWORD "
                        "ou la section [admin] des secrets Streamlit."
                    )

def show_admin_page():
    """Espace Administration & Pilotage Agronomique Ultra-Design."""
    from pathlib import Path

    st.warning(
        "Données de démonstration : carte, annuaire, alertes et graphiques sont statiques "
        "et ne décrivent pas des exploitations réelles."
    )
    try:
        metrics = load_test_metrics(
            Path(__file__).resolve().parent / "reports" / "metrics.json"
        )
    except (OSError, ValueError) as exc:
        st.error(f"Le rapport de métriques est invalide : {exc}")
        metrics = None
    test_accuracy = (
        f"{metrics['test']['accuracy']:.1%}"
        if metrics is not None
        else "Non évaluée"
    )
    st.info(
        "Modèle entraîné chargé." if predictor.model_loaded
        else "Aucun modèle entraîné valide n'est chargé."
    )

    # --- TOP BANNER ADMIN CORPORATE ---
    st.markdown("""<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 28px 32px; border-radius: 20px; color: white; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15); border: 1px solid #334155;">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
<div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
<span style="background-color: #16A34A; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">Centre de Commande IA</span>
<span style="color: #94A3B8; font-size: 0.85rem;">Données de démonstration</span>
</div>
<h1 style="font-family: Outfit; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0; letter-spacing: -0.5px;">Supervision & Pilotage des Coopératives</h1>
<p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 1rem;">Cartographie décisionnelle parcelles, détection précoce des épidémies et gouvernance des algorithmes.</p>
</div>
<div style="background: rgba(255,255,255,0.08); backdrop-filter: blur(10px); padding: 14px 22px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12); text-align: right;">
<div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Statut Serveurs IA</div>
<div style="font-family: Outfit; font-size: 1.15rem; font-weight: 700; color: #4ADE80; margin-top: 2px;">État affiché ci-dessous</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # --- BANNIÈRE DE METRIQUES KPI ---
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown("""<div class="admin-kpi-card">
<div class="admin-kpi-label">Exploitations — démonstration</div>
<div class="admin-kpi-val">Données statiques</div>
</div>""", unsafe_allow_html=True)
    with col_k2:
        st.markdown("""<div class="admin-kpi-card blue">
<div class="admin-kpi-label">Diagnostics — démonstration</div>
<div class="admin-kpi-val">Données statiques</div>
</div>""", unsafe_allow_html=True)
    with col_k3:
        st.markdown("""<div class="admin-kpi-card amber">
<div class="admin-kpi-label">Vigilances — démonstration</div>
<div class="admin-kpi-val">Données statiques</div>
</div>""", unsafe_allow_html=True)
    with col_k4:
        st.markdown(f"""<div class="admin-kpi-card purple">
<div class="admin-kpi-label">Accuracy du jeu de test</div>
<div class="admin-kpi-val">{test_accuracy}</div>
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
<strong>Exemple d’alerte — données de démonstration :</strong> cette alerte régionale est statique et n'est pas issue d'une mesure.
</div>""", unsafe_allow_html=True)
        
        st.markdown("#### Publier une Notification Sanitaire Ciblée")
        with st.form("alert_form"):
            alert_zone = st.selectbox("Secteur agricole concerné", ["Tous les secteurs", "Kindia", "Labé", "Mamou", "Kankan", "Nzérékoré"])
            alert_disease = st.selectbox("Maladie ou symptôme identifié", CROPS)
            alert_severity = st.radio("Seuil d'urgence", ["Vigilance Jaune", "Alerte Rouge - Intervention Immédiate"], horizontal=True)
            alert_msg = st.text_area("Consignes de traitement & biocontrôle", "Humidité foliaire supérieure à 80%. Appliquer une pulvérisation de produit cuivré préventif sous 24 heures.")
            
            submit_alert = st.form_submit_button("Diffuser l'Alerte aux Agriculteurs de la Zone", type="primary")
            if submit_alert:
                st.info(
                    "Aucune notification n'a été envoyée : le portail ne dispose pas "
                    "d'un service de diffusion connecté."
                )

    # --- TAB 4 : SUPERVISION DES MODELES IA ---
    with tab_ai:
        st.markdown("<h3 style='font-family: Outfit; color: #0F172A;'>Supervision des Algorithmes & Modèles IA</h3>", unsafe_allow_html=True)
        
        col_ai1, col_ai2 = st.columns(2, gap="large")
        with col_ai1:
            st.markdown("#### État réel du modèle")
            st.write(
                "Un modèle entraîné est chargé."
                if predictor.model_loaded
                else "Aucun modèle entraîné valide n'est disponible."
            )
            st.write(f"Accuracy mesurée sur le jeu de test : {test_accuracy}.")
            if metrics is not None:
                st.caption("Source : reports/metrics.json — jeu de test réservé.")
            
        with col_ai2:
            st.markdown("#### Préparation et entraînement")
            st.write(
                "Le téléchargement et l'entraînement ne sont pas déclenchés depuis "
                "l'interface. Ils nécessitent un jeu de données vérifié et une validation "
                "de licence ; consultez scripts/prepare_dataset.py et train_model.py."
            )
            st.caption(
                "Un modèle ne peut pas être remplacé par un fichier isolé : "
                "l'artefact et ses métadonnées de classes doivent correspondre."
            )

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