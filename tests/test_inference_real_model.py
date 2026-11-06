import json
from pathlib import Path

import pytest
from PIL import Image

from src.models.predictor import DiseasePredictor


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src" / "models" / "trained_models" / "farmview_model.keras"
CLASSES_PATH = ROOT / "src" / "models" / "trained_models" / "farmview_model.classes.json"
SAMPLES_PATH = ROOT / "data" / "demo_samples"


def has_loadable_model():
    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        return False
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH)
        metadata = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        return model.output_shape[-1] == len(metadata["classes"])
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not has_loadable_model(),
    reason="Un modèle .keras entraîné et ses classes compatibles ne sont pas disponibles.",
)


def test_real_model_predicts_every_demo_class():
    samples = sorted(SAMPLES_PATH.glob("*/*"))
    assert samples, "Aucune image de démonstration étiquetée n'est disponible."

    predictor = DiseasePredictor(MODEL_PATH)
    assert predictor.model_loaded

    for sample_path in samples:
        expected_class = sample_path.parent.name
        result = predictor.predict(Image.open(sample_path))
        assert result["source"] == "trained_model"
        assert result["predicted_class"] == expected_class
        assert result["diagnostic_status"] == (
            "healthy" if expected_class.endswith("___healthy") else "disease_detected"
        )
