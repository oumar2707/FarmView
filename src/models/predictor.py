"""Inference for a trained FarmView MobileNetV2 model."""

import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.disease_info import DISEASE_INFO
from src.utils.image_processor import preprocess_mobilenet_v2

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """Load a Keras model only when its class metadata is compatible."""

    def __init__(self, model_path=None):
        self.model_loaded = False
        self.tf_model = None
        self.class_names = []
        self.available_classes = list(DISEASE_INFO)

        if model_path:
            self.model_paths = [Path(model_path)]
        else:
            base_dir = Path(__file__).resolve().parents[2]
            model_dir = base_dir / "src" / "models" / "trained_models"
            self.model_paths = [
                model_dir / "farmview_model.keras",
                model_dir / "farmview_model.h5",
            ]

        self.model_info = {
            "status": "model_unavailable",
            "version": None,
            "accuracy": None,
            "classes": 0,
            "engine": None,
            "source": "none",
        }
        self.load_model()

    @staticmethod
    def _classes_metadata_path(model_path):
        return model_path.with_name(f"{model_path.stem}.classes.json")

    @staticmethod
    def _read_class_names(metadata_path):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = metadata.get("classes") if isinstance(metadata, dict) else None
        if (
            not isinstance(class_names, list)
            or not class_names
            or any(not isinstance(name, str) or not name for name in class_names)
            or len(class_names) != len(set(class_names))
        ):
            raise ValueError("Le fichier de classes est vide ou invalide.")
        if any(name not in DISEASE_INFO for name in class_names):
            raise ValueError("Le fichier contient une classe absente du catalogue phytosanitaire.")
        return class_names

    @staticmethod
    def _output_class_count(model):
        output_shape = model.output_shape
        if isinstance(output_shape, list):
            if len(output_shape) != 1:
                raise ValueError("Le modèle doit avoir une unique sortie de classification.")
            output_shape = output_shape[0]
        if not output_shape or output_shape[-1] is None:
            raise ValueError("La dimension de sortie du modèle est indéterminée.")
        return int(output_shape[-1])

    def load_model(self):
        """Load the first valid Keras artifact with matching class metadata."""
        self.model_loaded = False
        self.tf_model = None
        self.class_names = []
        last_error = "Aucun fichier modèle entraîné n'a été trouvé."

        for path in self.model_paths:
            if not path.is_file():
                continue
            try:
                classes_path = self._classes_metadata_path(path)
                if not classes_path.is_file():
                    raise ValueError(f"Métadonnées de classes absentes : {classes_path}")
                class_names = self._read_class_names(classes_path)

                import tensorflow as tf

                model = tf.keras.models.load_model(path)
                if self._output_class_count(model) != len(class_names):
                    raise ValueError(
                        "Le nombre de sorties du modèle ne correspond pas au fichier de classes."
                    )

                self.tf_model = model
                self.class_names = class_names
                self.model_loaded = True
                self.model_info = {
                    "status": "trained_model",
                    "version": None,
                    "accuracy": None,
                    "classes": len(class_names),
                    "engine": "TensorFlow / Keras",
                    "source": "trained_model",
                    "path": str(path),
                }
                logger.info("Modèle compatible chargé : %s", path)
                return True
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Modèle ignoré (%s) : %s", path, exc)

        self.model_info.update({
            "status": "model_unavailable",
            "classes": 0,
            "engine": None,
            "source": "none",
            "error": last_error,
        })
        return False

    @staticmethod
    def _as_rgb_image(image_input):
        if image_input is None:
            raise ValueError("Aucune image n'a été fournie.")
        if isinstance(image_input, (str, os.PathLike)) or hasattr(image_input, "read"):
            with Image.open(image_input) as image:
                return image.convert("RGB")
        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("L'image est vide.")
            return Image.fromarray(np.asarray(image_input).astype(np.uint8)).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise TypeError(f"Type d'image non pris en charge : {type(image_input)!r}")

    def _unavailable_result(self, status, message):
        return {
            "predicted_class": None,
            "confidence": None,
            "top_predictions": [],
            "diagnostic_status": status,
            "source": "none",
            "message": message,
            "model_info": self.model_info,
        }

    def predict(self, image_input):
        """Predict from a trained model or explicitly report a non-diagnostic state."""
        image = self._as_rgb_image(image_input)
        if not self.model_loaded or self.tf_model is None:
            return self._unavailable_result(
                "model_unavailable",
                "Aucun modèle entraîné et compatible n'est disponible.",
            )

        try:
            raw_batch = np.expand_dims(
                np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0
            )
            import tensorflow as tf

            batch = preprocess_mobilenet_v2(
                raw_batch,
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
            output = np.asarray(self.tf_model.predict(batch, verbose=0))
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("Le modèle a renvoyé une forme de sortie invalide.")

            scores = output[0]
            if len(scores) != len(self.class_names):
                raise ValueError("Les sorties du modèle ne correspondent pas aux classes.")
            if (
                not np.all(np.isfinite(scores))
                or np.any(scores < 0)
                or np.any(scores > 1)
                or not np.isclose(float(scores.sum()), 1.0, atol=1e-3)
            ):
                raise ValueError("Le modèle a renvoyé des scores Softmax invalides.")

            indices = np.argsort(scores)[::-1][:3]
            top_predictions = [
                (self.class_names[index], float(scores[index])) for index in indices
            ]
            predicted_class, confidence = top_predictions[0]
            if confidence < 0.75:
                status = "uncertain"
            elif predicted_class.endswith("___healthy"):
                status = "healthy"
            else:
                status = "disease_detected"

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top_predictions": top_predictions,
                "diagnostic_status": status,
                "source": "trained_model",
                "message": None,
                "model_info": self.model_info,
            }
        except Exception as exc:
            logger.exception("Erreur pendant l'inférence du modèle")
            return self._unavailable_result(
                "inference_error", f"L'inférence du modèle a échoué : {exc}"
            )

    def get_model_info(self):
        return self.model_info


predictor = DiseasePredictor()
