"""Prepare a deduplicated PlantVillage subset after manual license review."""

import argparse
import hashlib
import io
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "dataset"
DEFAULT_DEMO_OUTPUT = ROOT / "data" / "demo_samples"
TFDS_NAME = "plant_village/1.0.2"
REQUIRED_SOURCE_CLASSES = {
    "Apple___Apple_scab": "Apple___Apple_scab",
    "Apple___Black_rot": "Apple___Black_rot",
    "Apple___healthy": "Apple___healthy",
    "Corn___Common_rust": "Corn___Common_rust",
    "Potato___Late_blight": "Potato___Late_blight",
    "Tomato___Early_blight": "Tomato___Early_blight",
}
OPTIONAL_SOURCE_CLASSES = {
    "Grape___Esca": "Grape___Esca_(Black_Measles)",
}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def map_source_classes(source_class_names):
    """Map available PlantVillage labels to documented FarmView class identifiers."""
    available = set(source_class_names)
    missing = sorted(set(REQUIRED_SOURCE_CLASSES.values()) - available)
    if missing:
        raise ValueError(f"Classes PlantVillage obligatoires absentes : {missing}")

    mapping = dict(REQUIRED_SOURCE_CLASSES)
    mapping.update(
        {
            farmview_class: source_class
            for farmview_class, source_class in OPTIONAL_SOURCE_CLASSES.items()
            if source_class in available
        }
    )
    return dict(sorted(mapping.items()))


def image_digest(image):
    """Hash decoded RGB pixels so differently encoded exact image copies are detected."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def split_examples(examples, seed):
    """Split one class deterministically into disjoint train/validation/test lists."""
    if len(examples) < 300 or len(examples) > 500:
        raise ValueError(
            f"Chaque classe doit contenir 300 à 500 images uniques; obtenu: {len(examples)}."
        )

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * SPLIT_RATIOS["train"])
    val_end = train_end + int(len(shuffled) * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def require_license_confirmation(license_name, verification_evidence, confirmed):
    """Prevent downloading or deriving a manifest until a human confirms dataset rights."""
    if not confirmed:
        raise ValueError(
            "Usage bloqué : vérifiez la licence PlantVillage, puis fournissez "
            "--license-confirmed, --license-name et --verification-evidence."
        )
    if not license_name.strip() or not verification_evidence.strip():
        raise ValueError("Le nom de licence et la preuve de vérification sont obligatoires.")


def collect_examples(tfds, output_classes, images_per_class):
    """Read selected original TFDS labels and deduplicate exact images across classes."""
    builder = tfds.builder(TFDS_NAME)
    info = builder.info
    source_names = info.features["label"].names
    class_mapping = map_source_classes(source_names)

    if not 300 <= images_per_class <= 500:
        raise ValueError("--images-per-class doit être compris entre 300 et 500.")

    source_to_farmview = {
        source_name: farmview_name for farmview_name, source_name in class_mapping.items()
    }
    selected_source_names = set(source_to_farmview)
    examples = {name: {} for name in output_classes}
    seen_hashes = {}
    source = tfds.load(TFDS_NAME, split="train", shuffle_files=False)

    for record in tfds.as_numpy(source):
        source_name = source_names[int(record["label"])]
        if source_name not in selected_source_names:
            continue

        farmview_name = source_to_farmview[source_name]
        image = Image.fromarray(record["image"]).convert("RGB")
        digest = image_digest(image)
        previous_class = seen_hashes.get(digest)
        if previous_class is not None and previous_class != farmview_name:
            raise ValueError(
                f"Image dupliquée entre les classes {previous_class} et {farmview_name}."
            )
        seen_hashes[digest] = farmview_name
        examples[farmview_name].setdefault(digest, image)

    result = {}
    for class_index, class_name in enumerate(output_classes):
        unique_examples = sorted(examples[class_name].items())
        if len(unique_examples) < 300:
            raise ValueError(
                f"Classe {class_name}: seulement {len(unique_examples)} images uniques; "
                "minimum requis: 300."
            )
        chooser = random.Random(2026 + class_index)
        selected = chooser.sample(unique_examples, min(images_per_class, len(unique_examples)))
        result[class_name] = [
            {"digest": digest, "image": image} for digest, image in selected
        ]
    return result, class_mapping


def prepare_dataset(
    output_dir,
    demo_dir,
    license_name,
    verification_evidence,
    license_confirmed=False,
    images_per_class=400,
    seed=2026,
):
    """Download and prepare a verified subset; refuses to overwrite an existing dataset."""
    require_license_confirmation(license_name, verification_evidence, license_confirmed)
    output_dir = Path(output_dir)
    demo_dir = Path(demo_dir)
    if any((output_dir / split).exists() for split in SPLIT_RATIOS):
        raise FileExistsError(
            f"Un découpage existe déjà dans {output_dir}; déplacez-le manuellement avant préparation."
        )
    if demo_dir.exists() and any(demo_dir.iterdir()):
        raise FileExistsError(
            f"Le dossier de démonstration n'est pas vide : {demo_dir}; aucun fichier ne sera remplacé."
        )

    import tensorflow_datasets as tfds

    mapping = map_source_classes(tfds.builder(TFDS_NAME).info.features["label"].names)
    class_names = sorted(mapping)
    if not 300 <= images_per_class <= 500:
        raise ValueError("--images-per-class doit être compris entre 300 et 500.")
    examples_by_class, mapping = collect_examples(tfds, class_names, images_per_class)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    demo_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="farmview-prepare-") as temp_name:
        staging = Path(temp_name)
        for split_name in SPLIT_RATIOS:
            (staging / "dataset" / split_name).mkdir(parents=True)
        (staging / "demo").mkdir(parents=True)

        split_counts = {split_name: {} for split_name in SPLIT_RATIOS}
        for class_index, class_name in enumerate(class_names):
            splits = split_examples(examples_by_class[class_name], seed + class_index)
            for split_name, split_examples_for_class in splits.items():
                class_dir = staging / "dataset" / split_name / class_name
                class_dir.mkdir()
                for index, example in enumerate(split_examples_for_class):
                    image_path = class_dir / f"leaf_{index:04d}_{example['digest'][:8]}.jpg"
                    example["image"].save(image_path, format="JPEG", quality=95)
                split_counts[split_name][class_name] = len(split_examples_for_class)

            demo_example = splits["val"][0]
            demo_class_dir = staging / "demo" / class_name
            demo_class_dir.mkdir()
            demo_example["image"].save(
                demo_class_dir / "example.jpg", format="JPEG", quality=95
            )

        manifest = {
            "verified": True,
            "source": f"TensorFlow Datasets {TFDS_NAME}; underlying source: Mendeley Data tywbtsjrjv/1",
            "license": license_name,
            "verification_evidence": verification_evidence,
            "license_confirmed_by_operator": True,
            "classes": class_names,
            "source_class_mapping": mapping,
            "images_per_class_max": images_per_class,
            "split_ratios": SPLIT_RATIOS,
            "split_counts": split_counts,
            "seed": seed,
            "deduplication": "SHA-256 of decoded RGB shape and pixel data; duplicates across selected classes rejected",
            "note": (
                "verified records a manual operator attestation, not independent legal review. "
                "TFDS and Mendeley catalog pages did not state the dataset reuse license."
            ),
        }
        manifest_path = staging / "dataset" / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in SPLIT_RATIOS:
            shutil.move(str(staging / "dataset" / split_name), output_dir / split_name)
        shutil.move(str(manifest_path), output_dir / "dataset_manifest.json")
        demo_dir.mkdir(parents=True, exist_ok=True)
        for class_dir in (staging / "demo").iterdir():
            shutil.move(str(class_dir), demo_dir / class_dir.name)

    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_OUTPUT)
    parser.add_argument("--images-per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--license-confirmed", action="store_true")
    parser.add_argument("--license-name", default="")
    parser.add_argument("--verification-evidence", default="")
    args = parser.parse_args(argv)

    try:
        manifest = prepare_dataset(
            output_dir=args.output_dir,
            demo_dir=args.demo_dir,
            license_name=args.license_name,
            verification_evidence=args.verification_evidence,
            license_confirmed=args.license_confirmed,
            images_per_class=args.images_per_class,
            seed=args.seed,
        )
    except Exception as exc:
        parser.exit(1, f"Préparation annulée : {exc}\n")
    print(f"Dataset préparé: {args.output_dir} ({len(manifest['classes'])} classes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
