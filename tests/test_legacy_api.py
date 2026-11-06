from PIL import Image

from model.model import predict_disease


def test_legacy_api_never_invents_a_disease(tmp_path):
    image_path = tmp_path / "leaf.jpg"
    Image.new("RGB", (20, 20), "green").save(image_path)

    result = predict_disease(image_path)

    assert result["diagnostic_status"] == "model_unavailable"
    assert result["source"] == "none"
    assert result["maladie"] is None
