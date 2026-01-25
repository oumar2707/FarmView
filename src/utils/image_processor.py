"""
Module de prétraitement des images pour le modèle IA.
"""

import numpy as np
from PIL import Image
import cv2

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Charge et prétraite une image pour la prédiction.
    
    Args:
        image_path: Chemin vers l'image ou objet image
        target_size: Taille cible pour le modèle (224x224 pour MobileNetV2)
    
    Returns:
        image_array: Image prétraitée normalisée
    """
    # Si c'est un chemin de fichier
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner
    image = image.resize(target_size)
    
    # Convertir en array numpy
    image_array = np.array(image)
    
    # Normaliser les pixels [0, 255] -> [0, 1]
    image_array = image_array / 255.0
    
    # Ajouter une dimension pour le batch (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def resize_image(image, max_size=800):
    """
    Redimensionne une image pour l'affichage tout en gardant les proportions.
    """
    # Convertir PIL Image à OpenCV si nécessaire
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]
    
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    return image