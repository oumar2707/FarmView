import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.image_processor import preprocess_mobilenet_v2
from train_model import _preprocess_dataset, validate_dataset_manifest


class FakeDataset:
    def __init__(self, value=None):
        self.value = value
        self.map_calls = 0

    def map(self, function, num_parallel_calls=None):
        self.map_calls += 1
        return function(np.array([0.0, 127.5, 255.0], dtype=np.float32), np.array([1]))


def test_shared_mobilenet_v2_preprocessing_contract():
    seen = []

    def fake_preprocess(values):
        seen.append(values.copy())
        return values / 127.5 - 1.0

    raw_rgb = np.array([[[[0.0, 127.5, 255.0]]]], dtype=np.float32)
    result = preprocess_mobilenet_v2(raw_rgb, fake_preprocess)

    np.testing.assert_array_equal(seen[0], raw_rgb)
    np.testing.assert_allclose(result, [[[[-1.0, 0.0, 1.0]]]])


def test_training_preprocess_augments_only_when_requested():
    preprocess_calls = []
    augmentation_calls = []

    def preprocess(values):
        preprocess_calls.append(values.copy())
        return values / 127.5 - 1.0

    def augmentation(values, training):
        augmentation_calls.append(training)
        return values + 1

    tf = SimpleNamespace(data=SimpleNamespace(AUTOTUNE=1))
    train_dataset = FakeDataset()
    _preprocess_dataset(train_dataset, preprocess, augmentation, tf)
    val_dataset = FakeDataset()
    _preprocess_dataset(val_dataset, preprocess, tf=tf)

    assert augmentation_calls == [True]
    assert train_dataset.map_calls == 1
    assert val_dataset.map_calls == 1
    assert len(preprocess_calls) == 2


def test_training_refuses_missing_or_unverified_manifest(tmp_path):
    with pytest.raises(ValueError, match="manifeste absent"):
        validate_dataset_manifest(tmp_path)

    manifest = {
        "verified": False,
        "source": "source",
        "license": "license",
        "verification_evidence": "manual",
        "license_confirmed_by_operator": False,
        "classes": ["Apple___healthy"],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="validation manuelle"):
        validate_dataset_manifest(tmp_path)


def test_training_refuses_split_or_manifest_class_mismatch(tmp_path):
    classes = ["Apple___healthy"]
    manifest = {
        "verified": True,
        "source": "source",
        "license": "license",
        "verification_evidence": "manually checked source license",
        "license_confirmed_by_operator": True,
        "classes": classes,
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for split in ("train", "val", "test"):
        (tmp_path / split / "Other___class").mkdir(parents=True)
        (tmp_path / split / "Other___class" / "sample.jpg").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="ne correspondent pas"):
        validate_dataset_manifest(tmp_path)
