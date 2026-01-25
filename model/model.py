import tensorflow as tf
import numpy as np
from PIL import Image
MALADIES_FR = {
    "mildew": "Mildiou",
    "powdery_mildew": "Oïdium",
    "rust": "Rouille",
    "Mildiou": "Mildiou",
    "Oidium": "Oïdium",
    "Oïdium": "Oïdium",
    "Rouille": "Rouille"
}
MALADIES_INFO = {
    "Mildiou": {
        "description": "Maladie fongique très répandue qui provoque des taches jaunes ou brunes sur les feuilles et peut détruire toute la culture.",
        "conseils": "Éviter l'excès d'humidité, utiliser des fongicides adaptés, retirer les feuilles infectées.",
        "niveau": "Élevé"
    },
    "Oïdium": {
        "description": "Maladie causée par un champignon qui apparaît sous forme de poudre blanche sur les feuilles.",
        "conseils": "Améliorer l'aération, éviter l'arrosage excessif, appliquer du soufre ou un traitement naturel.",
        "niveau": "Moyen"
    },
    "Rouille": {
        "description": "Maladie fongique reconnaissable par des taches orangées ou brunes sur les feuilles.",
        "conseils": "Supprimer les feuilles atteintes, espacer les plants, utiliser un traitement fongicide.",
        "niveau": "Moyen"
    }
}




def load_image(image_path, img_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_disease(image_path):
    image = load_image(image_path)

    classes = ["Feuille saine", "Mildiou", "Oïdium", "Rouille"]
    prediction = np.random.choice(classes)

    # Si la feuille est saine
    if prediction == "Feuille saine":
        return {
            "maladie": "Feuille saine",
            "description": "La feuille est en bon état et ne présente aucun signe de maladie.",
            "conseils": "Aucune action nécessaire. Continuer une bonne pratique agricole.",
            "niveau": "Aucun"
        }

    # Infos maladie
    resultat = MALADIES_INFO.get(prediction, {
        "description": "Maladie inconnue.",
        "conseils": "Aucun conseil disponible.",
        "niveau": "Inconnu"
    })

    return {
        "maladie": prediction,
        "description": resultat["description"],
        "conseils": resultat["conseils"],
        "niveau": resultat["niveau"]
    }
