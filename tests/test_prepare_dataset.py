import numpy as np
import pytest
from PIL import Image

from scripts.prepare_dataset import (
    image_digest,
    map_source_classes,
    require_license_confirmation,
    split_examples,
)


SOURCE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Common_rust",
    "Potato___Late_blight",
    "Tomato___Early_blight",
    "Grape___Esca_(Black_Measles)",
]


def test_license_gate_requires_operator_confirmation_and_evidence():
    with pytest.raises(ValueError, match="Usage bloqué"):
        require_license_confirmation("", "", False)
    with pytest.raises(ValueError, match="obligatoires"):
        require_license_confirmation("", "", True)


def test_license_gate_accepts_explicit_operator_attestation():
    require_license_confirmation("License to be verified", "reviewed URL", True)


def test_source_class_mapping_normalizes_grape_label():
    mapping = map_source_classes(SOURCE_CLASSES)

    assert mapping["Grape___Esca"] == "Grape___Esca_(Black_Measles)"
    assert list(mapping) == sorted(mapping)


def test_source_class_mapping_allows_absent_optional_grape():
    mapping = map_source_classes(SOURCE_CLASSES[:-1])

    assert "Grape___Esca" not in mapping


def test_source_class_mapping_rejects_missing_required_classes():
    with pytest.raises(ValueError, match="obligatoires absentes"):
        map_source_classes(SOURCE_CLASSES[1:])


def test_class_splits_are_deterministic_disjoint_and_70_15_15():
    items = list(range(400))

    first = split_examples(items, seed=19)
    second = split_examples(items, seed=19)

    assert first == second
    assert [len(first[key]) for key in ("train", "val", "test")] == [280, 60, 60]
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["val"]) & set(first["test"]))


def test_split_rejects_unreasonable_class_size():
    with pytest.raises(ValueError, match="300 à 500"):
        split_examples(list(range(299)), seed=19)


def test_image_digest_detects_identical_rgb_pixels():
    pixels = np.full((16, 16, 3), 127, dtype=np.uint8)
    assert image_digest(Image.fromarray(pixels)) == image_digest(Image.fromarray(pixels.copy()))
