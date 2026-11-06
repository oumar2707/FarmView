import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / "data" / "demo_samples"


def demo_images():
    return [path for path in SAMPLES_PATH.glob("*/*") if path.is_file()]


@pytest.mark.skipif(not demo_images(), reason="Les images démo validées ne sont pas encore préparées.")
def test_demo_images_have_no_cross_class_exact_duplicates():
    digest_to_class = {}

    for image_path in demo_images():
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        class_name = image_path.parent.name
        previous_class = digest_to_class.setdefault(digest, class_name)
        assert previous_class == class_name, (
            f"Le même fichier image apparaît dans les classes {previous_class} et {class_name}."
        )


PREPARED_SPLITS = ROOT / "data" / "dataset"


@pytest.mark.skipif(
    not any((PREPARED_SPLITS / split).is_dir() for split in ("train", "val", "test")),
    reason="Les découpages PlantVillage validés ne sont pas encore préparés.",
)
def test_prepared_images_have_no_cross_class_exact_duplicates():
    digest_to_class = {}

    for split_dir in (PREPARED_SPLITS / "train", PREPARED_SPLITS / "val", PREPARED_SPLITS / "test"):
        if not split_dir.is_dir():
            continue
        for image_path in (path for path in split_dir.rglob("*") if path.is_file()):
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            class_name = image_path.parent.name
            previous_class = digest_to_class.setdefault(digest, class_name)
            assert previous_class == class_name, (
                f"Le même fichier image apparaît dans les classes {previous_class} et {class_name}."
            )
