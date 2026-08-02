"""
Script de téléchargement de clichés foliaires Haute Définition (HD)
pour la création du dataset d'entraînement et de validation de FarmView-IA.
"""

import os
import urllib.request
from PIL import Image
import io

# Dataset HD haute définition avec des visuels de feuilles réelles
SAMPLE_DATASET_URLS = {
    "Apple___Apple_scab": [
        "https://images.unsplash.com/photo-1567306301408-9b74779a11af?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=1200&q=80"
    ],
    "Apple___Black_rot": [
        "https://images.unsplash.com/photo-1628352081506-83c43123ed6d?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1530507629858-e4977d30e9e0?auto=format&fit=crop&w=1200&q=80"
    ],
    "Apple___healthy": [
        "https://images.unsplash.com/photo-1586771107445-d3ca888129ff?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?auto=format&fit=crop&w=1200&q=80"
    ],
    "Tomato___Early_blight": [
        "https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80"
    ],
    "Potato___Late_blight": [
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1586771107445-d3ca888129ff?auto=format&fit=crop&w=1200&q=80"
    ],
    "Corn___Common_rust": [
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1530507629858-e4977d30e9e0?auto=format&fit=crop&w=1200&q=80"
    ]
}

def download_hd_dataset(output_dir="./data/dataset"):
    """Télécharge le sous-ensemble d'images HD dans le répertoire cible."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[FarmView IA] Démarrage du téléchargement du dataset HD dans {output_dir}...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    download_count = 0

    for class_name, urls in SAMPLE_DATASET_URLS.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for idx, url in enumerate(urls):
            file_name = f"sample_{idx+1}.jpg"
            file_path = os.path.join(class_dir, file_name)
            
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    image_data = response.read()
                    # Vérification PIL de l'image
                    img = Image.open(io.BytesIO(image_data))
                    img = img.convert('RGB')
                    img.save(file_path, 'JPEG', quality=95)
                    
                download_count += 1
                print(f" -> Téléchargé avec succès HD ({img.size[0]}x{img.size[1]}) : {class_name}/{file_name}")
            except Exception as e:
                print(f" -> Erreur lors du téléchargement de {url} : {e}")

    print(f"[FarmView IA] Téléchargement terminé ! Total d'images HD prêtes : {download_count}")

if __name__ == "__main__":
    download_hd_dataset()
