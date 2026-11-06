"""Google Colab entry point; requires explicit manual dataset-license review."""

import argparse
from pathlib import Path

from scripts.prepare_dataset import DEFAULT_DEMO_OUTPUT, DEFAULT_OUTPUT, prepare_dataset
from train_model import train_farmview_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license-confirmed", action="store_true")
    parser.add_argument("--license-name", default="")
    parser.add_argument("--verification-evidence", default="")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_OUTPUT)
    parser.add_argument("--images-per-class", type=int, default=400)
    args = parser.parse_args(argv)

    prepare_dataset(
        output_dir=args.dataset_dir,
        demo_dir=args.demo_dir,
        license_name=args.license_name,
        verification_evidence=args.verification_evidence,
        license_confirmed=args.license_confirmed,
        images_per_class=args.images_per_class,
    )
    model_path = train_farmview_model(dataset_dir=args.dataset_dir)
    print(f"Modèle .keras exporté : {model_path}")
    print(f"Métadonnées des classes : {model_path.with_name('farmview_model.classes.json')}")
    print("Métriques et matrice de confusion : reports/")


if __name__ == "__main__":
    main()
