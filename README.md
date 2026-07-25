<div align="center">

  # 🌱 FarmView-IA

  ### **Plateforme Intelligente de Diagnostic Phytosanitaire & Surveillance Agronomique**

  [![Python](https://img.shields.io/badge/Python-3.10%2B-16A34A?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
  [![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
  [![Précision IA](https://img.shields.io/badge/Précision_IA-96.5%25-16A34A?style=for-the-badge)](https://github.com/)

  <p align="center">
    <strong>FarmView-IA</strong> est une solution logicielle d'agtech avancée combinant la <strong>Vision par Ordinateur (Deep Learning)</strong> et la <strong>Cartographie SIG</strong> pour permettre la détection précoce des pathologies végétales, la prévention des épidémies et l'optimisation des conseils en biocontrôle.
  </p>

  ---

</div>

## 🌟 Sommaire

- [À propos du projet](#-à-propos-du-projet)
- [Fonctionnalités Clés](#-fonctionnalités-clés)
- [Architecture Technique](#-architecture-technique)
- [Structure du Répertoire](#-structure-du-répertoire)
- [Guide d'Installation](#-guide-dinstallation)
- [Guide d'Utilisation](#-guide-dutilisation)
- [Performances du Modèle IA](#-performances-du-modèle-ia)
- [Contribution & Licence](#-contribution--licence)

---

## 🍃 À propos du projet

Face aux défis climatiques et à la recrudescence des maladies foliaires (mildiou, tavelure, oïdium, etc.), les exploitants et coopératives agricoles nécessitent des outils décisionnels instantanés au champ.

**FarmView-IA** répond à cet enjeu en fournissant :
- **Un diagnostic instantané au champ** à partir d'une simple photographie de feuille.
- **Des protocoles de soins agro-écologiques** et de biocontrôle ciblé.
- **Un portail d'administration et de pilotage cartographique** pour superviser la santé parcellaire globale d'un réseau de coopératives.

---

## ⚡ Fonctionnalités Clés

### 👨‍🌾 Interface Publique & Exploitant
- **Diagnostic IA Instantané** : Téléversement de cliché foliaire avec calcul en direct de l'indice de certitude (`< 2s`).
- **Recommandations Phytosanitaires** : Traitements de biocontrôle agro-écologiques et options de traitement chimique modérées.
- **Base de Connaissances Illustrée** : Fiches synthétiques avec visuels HD des principales pathologies végétales.
- **Télémétrie & Risques Météo** : Indicateurs locaux de vent, d'humidité et d'évaluation des risques d'infection.

### 🛡️ Portail d'Administration & Pilotage Coopératives
- **Cartographie Géographique des Risques (Map SIG)** : Carte interactive des parcelles géolocalisées avec code couleur dynamique :
  - 🔴 **Alerte Rouge** : Foyer d'infection confirmé (ex: Mildiou à Kindia).
  - 🟡 **Vigilance Jaune** : Risque modéré nécessitant surveillance.
  - 🟢 **Vert** : Parcelle saine sous contrôle.
- **Tableau de Bord & KPIs** : Analytics en temps réel sur les volumes de diagnostics, les pathologies fréquentes et la répartition par bassin de production.
- **Diffusion d'Alertes Sanitaires** : Module d'émission de notifications et consignes phytosanitaires ciblées par secteur.
- **Supervision des Modèles IA** : Suivi des métriques (`Val Accuracy 96.5%`) et téléversement de nouvelles versions de modèles (`.h5` / `.keras`).
- **Annuaire des Coopérateurs** : Enregistrement et gestion du réseau d'exploitants agricoles.

---

## 🛠 Architecture Technique

| Composant | Technologie Utilisée | Description |
| :--- | :--- | :--- |
| **Interface Utilisateur** | Streamlit (Python) | Framework web interactif rapide et réactif |
| **Design système** | Vanilla CSS3 / HTML5 | Aesthetic White & Green Corporate avec typographies *Outfit* & *Inter* |
| **Moteur IA** | TensorFlow / Keras | CNN basé sur **MobileNetV2** (Transfer Learning) |
| **Traitement d'Image** | OpenCV / Pillow / NumPy | Prétraitement, redimensionnement et normalisation d'images |
| **Cartographie & Data** | Pandas / Streamlit Maps | Traitement des données GPS et rendu cartographique interactif |

---

## 📂 Structure du Répertoire

```text
FarmView-IA/
├── app.py                      # Application principale Streamlit (Routage & UI)
├── requirements.txt            # Dépendances Python du projet
├── README.md                   # Documentation officielle du dépôt
├── src/                        # Code source backend et modules IA
│   ├── data/
│   │   └── disease_info.py     # Base de données phytosanitaire & descriptions
│   ├── models/
│   │   ├── predictor.py        # Moteur de prédiction et inférence du modèle
│   │   └── trained_models/     # Stockage des fichiers modèles (.h5 / .keras)
│   └── utils/
│       └── image_processor.py  # Fonctions de prétraitement des visuels
└── assets/                     # Ressources visuelles et captures d'écran
```

---

## 🚀 Guide d'Installation

### Prérequis
- **Python 3.10** ou version supérieure
- **Git**

### 1. Cloner le dépôt GitHub
```bash
git clone https://github.com/VOTRE_NOM_UTILISATEUR/FarmView-IA.git
cd FarmView-IA
```

### 2. Créer et activer un environnement virtuel (recommandé)
```bash
# Sur Windows
python -m venv venv
.\venv\Scripts\activate

# Sur macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer l'application
```bash
streamlit run app.py
```
 L'application s'ouvrira automatiquement sur votre navigateur à l'adresse : `http://localhost:8501`.

---

## 💡 Guide d'Utilisation

1. **Lancer un Diagnostic** :
   - Naviguez vers l'onglet **Diagnostic**.
   - Sélectionnez éventuellement la culture concernée.
   - Glissez-déposez la photo d'une feuille présentant des symptômes.
   - Consultez l'analyse IA et les recommandations.

2. **Accéder à l'Espace Administration** :
   - Cliquez sur le bouton **Se Connecter** dans le menu supérieur.
   - Saisissez les identifiants administrateur de démonstration (`admin@farmview.ai` / `admin123`).
   - Accédez à la **Cartographie des Risques**, aux **KPIs** et à la **Supervision IA**.

---

## 📊 Performances du Modèle IA

Le modèle IA est entraîné sur le jeu de données universitaire PlantVillage (38 classes de symptômes foliaires).

| Métrique | Valeur |
| :--- | :--- |
| **Architecture Réseau** | MobileNetV2 (Feature Extractor + Classification Head) |
| **Précision globale (Validation Accuracy)** | **96.5%** |
| **Perte de validation (Validation Loss)** | **0.118** |
| **Temps d'inférence moyen** | **< 1.8 seconde** |
| **Format des entrées** | `224 x 224 x 3` (RGB Normalized) |

---

## 📄 Contribution & Licence

Les contributions sont les bienvenues ! Pour toute proposition d'amélioration ou signalement de bug, merci d'ouvrir une *Issue* ou de soumettre une *Pull Request*.

Ce projet est sous licence **MIT**. Consulter le fichier `LICENSE` pour plus de détails.

---

<div align="center">
  <p>Développé avec ❤️ pour l'innovation agricole et la protection des cultures.</p>
</div>
