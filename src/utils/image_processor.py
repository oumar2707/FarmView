"""
Module de prétraitement des images pour le modèle IA.
"""

import numpy as np
from PIL import Image
import cv2
import os


def load_and_preprocess_image(image_path, target_size=(224, 224), preprocess_input=None):
    """
    Load an image as RGB and apply the shared MobileNetV2 preprocessing.
    """
    if not isinstance(target_size, tuple) or len(target_size) != 2 or any(
        not isinstance(size, int) or size <= 0 for size in target_size
    ):
        raise ValueError("target_size doit contenir deux entiers positifs.")

    if isinstance(image_path, (str, os.PathLike)) or hasattr(image_path, "read"):
        with Image.open(image_path) as opened_image:
            opened_image.load()
            image = opened_image.convert("RGB")
    elif isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        raise TypeError("image_path doit être un chemin, un flux binaire ou une image PIL.")

    if image.width <= 0 or image.height <= 0:
        raise ValueError("L'image est vide.")

    image = image.resize(target_size)
    image_batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    return preprocess_mobilenet_v2(image_batch, preprocess_input)


def preprocess_mobilenet_v2(image_batch, preprocess_input=None):
    """Apply the MobileNetV2 preprocessing function to raw RGB values in [0, 255]."""
    if preprocess_input is None:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return preprocess_input(np.asarray(image_batch, dtype=np.float32))


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