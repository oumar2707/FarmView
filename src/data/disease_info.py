"""
Informations sur les maladies des plantes et base de données phytosanitaires.
Adapté au dataset universel PlantVillage / AgTech.
"""

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "name": "Tavelure du pommier",
        "crop": "Pommier",
        "description": "Maladie fongique grave causée par Venturia inaequalis. Provoque des taches d'aspect velouté olive à brun foncé sur les feuilles et les fruits.",
        "treatment": [
            "Ramasser et éliminer les feuilles tombées à l'automne pour réduire l'inoculum",
            "Pratiquer une taille d'éclaircissage pour favoriser la ventilation du feuillage",
            "Appliquer des traitements préventifs à base d'extraits de prêle ou de bicarbonate de potassium"
        ],
        "chemical": "Fongicides à base de soufre mouillable ou de cuivre (Bouillie bordelaise)"
    },
    "Apple___Black_rot": {
        "name": "Pourriture noire du pommier",
        "crop": "Pommier",
        "description": "Maladie fongique causée par Botryosphaeria obtusa. Se manifeste par des taches en œil de grenouille sur le feuillage et des chancres sur les branches.",
        "treatment": [
            "Tailler et brûler le bois mort ou chancré durant l'hiver",
            "Éviter les blessures d'écorce lors de la taille",
            "Assurer une nutrition équilibrée pour renforcer la défense naturelle"
        ],
        "chemical": "Traitement fongicide de synthèse à base de Myclobutanil ou Captane en cas de forte pression"
    },
    "Apple___healthy": {
        "name": "Pommier sain",
        "crop": "Pommier",
        "description": "Le feuillage présente une coloration verte homogène et une structure cellulaire vigoureuse sans aucun symptôme pathologique.",
        "treatment": [
            "Maintenir l'irrigation au pied et un paillage organique régulier",
            "Surveiller régulièrement l'apparition d'éventuels ravageurs (pucerons, acariens)"
        ],
        "chemical": "Aucun traitement nécessaire"
    },
    "Tomato___Early_blight": {
        "name": "Alternariose de la tomate",
        "crop": "Tomate",
        "description": "Maladie provoquée par Alternaria solani. Caractérisée par des taches brunes avec des cercles concentriques cernés de jaune sur les feuilles inférieures.",
        "treatment": [
            "Supprimer les feuilles basses touchées dès le début de l'infection",
            "Pailler le sol pour éviter les projections d'eau lors de l'arrosage",
            "Pratiquer une rotation des cultures de 3 ans minimum"
        ],
        "chemical": "Fongicides au cuivre ou Mancozèbe préventif"
    },
    "Potato___Late_blight": {
        "name": "Mildiou de la pomme de terre",
        "crop": "Pomme de terre",
        "description": "Infection dévastatrice causée par Phytophthora infestans. Taches d'aspect huileux et feutrage blanc au revers de la feuille par temps humide.",
        "treatment": [
            "Detruire immédiatement les plants fortement atteints",
            "Planter des tubercules certifiés indemnes de germes",
            "Aérer les rangs et limiter l'humidité stagnante"
        ],
        "chemical": "Fongicide systémique ou cuivre avant l'apparition des pluies"
    },
    "Corn___Common_rust": {
        "name": "Rouille commune du maïs",
        "crop": "Maïs",
        "description": "Causée par Puccinia sorghi. Formation de pustules allongées de couleur brun-poussière/orangée sur les deux faces du limbe.",
        "treatment": [
            "Utiliser des hybrides de maïs tolérants ou résistants",
            "Enfouir les résidus de culture après la récolte",
            "Planter tôt dans la saison"
        ],
        "chemical": "Fongicides triazoles ou strobilurines si l'infection survient avant la floraison"
    },
    "Grape___Esca": {
        "name": "Esca / Maladie du bois de la vigne",
        "crop": "Vigne",
        "description": "Complexe fongique provoquant une décoloration en 'peau de tigre' sur les feuilles et un dessèchement progressif des cepes.",
        "treatment": [
            "Protéger les plaies de taille avec du mastic cicatrisant",
            "Curer les cepes atteints ou les recéper",
            "Éliminer le bois mort du vignoble"
        ],
        "chemical": "Traitements à base de Trichoderma harzianum (Agent de biocontrôle)"
    }
}

CROPS = [
    "Pommier", "Tomate", "Pomme de terre", "Maïs", "Vigne",
    "Cerisier", "Raisin", "Pêcher", "Poivron", "Fraises"
]

def get_disease_info(class_name):
    """Retourne les informations complètes d'une maladie à partir de son nom de classe."""
    return DISEASE_INFO.get(class_name, {
        "name": "Symptôme non répertorié",
        "crop": "Inconnue",
        "description": "Symptômes visuels non concordants avec les souches connues. Nécessite une expertise agronomique complémentaire.",
        "treatment": [
            "Isoler le plant suspect de la parcelle",
            "Prendre une photo nette du limbe et consulter un spécialiste local"
        ],
        "chemical": "Consulter un conseiller phytosanitaire agréé"
    })