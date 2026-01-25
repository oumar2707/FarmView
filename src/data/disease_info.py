"""
Informations sur les maladies des plantes.
Ce module contient les labels, descriptions et recommandations de traitement.
"""

# Mapping des classes du modèle vers les noms de maladies
# Adapté au dataset PlantVillage
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "name": "Tavelure du pommier",
        "description": "Maladie fongique qui affecte les feuilles et les fruits.",
        "treatment": [
            "Supprimer les feuilles tombées à l'automne",
            "Appliquez un fongicide au printemps",
            "Choisir des variétés résistantes"
        ],
        "chemical": "Fongicides à base de soufre ou de cuivre"
    },
    "Apple___Black_rot": {
        "name": "Pourriture noire du pommier",
        "description": "Maladie causée par le champignon Botryosphaeria obtusa.",
        "treatment": [
            "Tailler les branches malades",
            "Éviter l'irrigation par aspersion",
            "Brûler les débris infectés"
        ],
        "chemical": "Fongicides à base de myclobutanil"
    },
    "Apple___healthy": {
        "name": "Pommier sain",
        "description": "La plante ne présente aucun signe de maladie visible.",
        "treatment": [
            "Continuer les bonnes pratiques culturales",
            "Surveiller régulièrement les cultures"
        ],
        "chemical": "Aucun traitement nécessaire"
    },
    # Nous ajouterons plus de maladies plus tard
    # Pour l'instant, gardons-le simple
}

# Liste des cultures supportées
CROPS = [
    "Pommier", "Cerisier", "Maïs", "Raisin", "Pêcher",
    "Poivron", "Pomme de terre", "Framboisier", "Soja",
    "Citrouille", "Fraises", "Tomate"
]

def get_disease_info(class_name):
    """Retourne les informations d'une maladie à partir de son nom de classe."""
    return DISEASE_INFO.get(class_name, {
        "name": "Maladie inconnue",
        "description": "Information non disponible.",
        "treatment": ["Consultez un agronome local."],
        "chemical": "Non spécifié"
    })