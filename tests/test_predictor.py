import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image

from src.models.predictor import DiseasePredictor


CLASSES = ["Apple___healthy", "Apple___Apple_scab"]


class FakeModel:
    def __init__(self, scores=None, output_count=None, error=None):
        self.output_shape = (None, output_count or len(CLASSES))
        self.scores = scores if scores is not None else [[0.1, 0.9]]
        self.error = error

    def predict(self, batch, verbose=0):
        if self.error:
            raise self.error
        return np.asarray(self.scores)


def install_fake_tensorflow(monkeypatch, model):
    tensorflow = ModuleType("tensorflow")
    tensorflow.keras = SimpleNamespace(
        models=SimpleNamespace(load_model=lambda path: model),
        applications=SimpleNamespace(
            mobilenet_v2=SimpleNamespace(
                preprocess_input=lambda values: values / 127.5 - 1.0
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)


def make_model_artifacts(tmp_path, classes=CLASSES):
    model_path = tmp_path / "farmview_model.keras"
    model_path.write_bytes(b"mock model artifact")
    model_path.with_name("farmview_model.classes.json").write_text(
        json.dumps({"classes": classes}), encoding="utf-8"
    )
    return model_path


def test_absent_model_returns_model_unavailable(tmp_path):
    predictor = DiseasePredictor(tmp_path / "missing.keras")
    result = predictor.predict(Image.new("RGB", (16, 16)))

    assert not predictor.model_loaded
    assert result["diagnostic_status"] == "model_unavailable"
    assert result["predicted_class"] is None
    assert result["source"] == "none"


def test_missing_class_metadata_rejects_model(tmp_path, monkeypatch):
    model_path = tmp_path / "farmview_model.keras"
    model_path.write_bytes(b"mock model artifact")
    install_fake_tensorflow(monkeypatch, FakeModel())

    predictor = DiseasePredictor(model_path)

    assert not predictor.model_loaded
    assert predictor.model_info["status"] == "model_unavailable"


def test_class_metadata_and_model_output_must_match(tmp_path, monkeypatch):
    model_path = make_model_artifacts(tmp_path)
    install_fake_tensorflow(monkeypatch, FakeModel(output_count=3))

    predictor = DiseasePredictor(model_path)

    assert not predictor.model_loaded
    assert "ne correspond pas" in predictor.model_info["error"]


def test_model_prediction_is_labeled_as_trained_model(tmp_path, monkeypatch):
    model_path = make_model_artifacts(tmp_path)
    install_fake_tensorflow(monkeypatch, FakeModel())
    predictor = DiseasePredictor(model_path)

    result = predictor.predict(Image.new("L", (16, 16), color=100))

    assert result["source"] == "trained_model"
    assert result["predicted_class"] == "Apple___Apple_scab"
    assert result["diagnostic_status"] == "disease_detected"
    assert len(result["top_predictions"]) == 2


def test_low_confidence_prediction_is_uncertain(tmp_path, monkeypatch):
    model_path = make_model_artifacts(tmp_path)
    install_fake_tensorflow(monkeypatch, FakeModel(scores=[[0.6, 0.4]]))
    predictor = DiseasePredictor(model_path)

    result = predictor.predict(Image.new("RGB", (16, 16)))

    assert result["source"] == "trained_model"
    assert result["diagnostic_status"] == "uncertain"


def test_model_prediction_exception_returns_inference_error(tmp_path, monkeypatch):
    model_path = make_model_artifacts(tmp_path)
    install_fake_tensorflow(
        monkeypatch, FakeModel(error=RuntimeError("controlled prediction failure"))
    )
    predictor = DiseasePredictor(model_path)

    result = predictor.predict(Image.new("RGB", (16, 16)))

    assert result["diagnostic_status"] == "inference_error"
    assert result["source"] == "none"
    assert "controlled prediction failure" in result["message"]
