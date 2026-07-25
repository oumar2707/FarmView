"""
Module de prédiction des maladies.
Version de démonstration avec modèle factice.
À remplacer par un vrai modèle TensorFlow plus tard.
"""

import numpy as np
import random
from src.data.disease_info import DISEASE_INFO

class DiseasePredictor:
    """Classe pour la prédiction des maladies (version factice)."""
    
    def __init__(self, model_path=None):
        """
        Initialise le prédicteur.
        
        Args:
            model_path: Chemin vers le modèle (ignoré dans la version factice)
        """
        self.model_loaded = False
        self.available_classes = list(DISEASE_INFO.keys())
        
        # Informations de démonstration
        self.demo_info = {
            "status": "Démonstration",
            "version": "1.0.0",
            "accuracy": "94% (estimé)",
            "classes": len(self.available_classes)
        }
    
    def load_model(self):
        """Charge le modèle (simulation)."""
        import time
        time.sleep(0.5)  # Simulation du temps de chargement
        self.model_loaded = True
        return True
    
    def predict(self, image_array):
        """
        Prédit la maladie à partir d'une image.
        
        Args:
            image_array: Array numpy de l'image prétraitée
        
        Returns:
            dict: Résultats de la prédiction
        """
        if not self.model_loaded:
            self.load_model()
        
        # Simulation d'une prédiction aléatoire mais réaliste
        # Pour la démonstration, favorisons certaines classes communes
        
        # Classes communes pour la démonstration
        common_classes = [
            "Apple___Apple_scab",
            "Tomato___Early_blight", 
            "Potato___Late_blight",
            "Corn___Common_rust",
            "Grape___Black_rot"
        ]
        
        # Choisir une classe (70% chance d'une classe commune)
        if random.random() < 0.7:
            predicted_class = random.choice(common_classes)
        else:
            predicted_class = random.choice(self.available_classes)
        
        # Générer un score de confiance réaliste
        confidence = random.uniform(0.75, 0.98)
        
        # Générer des scores pour toutes les classes (simulation softmax)
        all_scores = {}
        for cls in self.available_classes:
            if cls == predicted_class:
                all_scores[cls] = confidence
            else:
                all_scores[cls] = (1 - confidence) / (len(self.available_classes) - 1)
        
        # Top 3 prédictions
        top_classes = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_scores,
            "top_predictions": top_classes,
            "model_info": self.demo_info
        }
    
    def get_model_info(self):
        """Retourne des informations sur le modèle."""
        return self.demo_info

# Instance globale pour l'application
predictor = DiseasePredictor()