"""Train, fine-tune, and evaluate MobileNetV2 using verified fixed data splits."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.utils.image_processor import preprocess_mobilenet_v2

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "data" / "dataset"
MODEL_DIR = ROOT / "src" / "models" / "trained_models"
REPORT_DIR = ROOT / "reports"
SPLITS = ("train", "val", "test")


def validate_dataset_manifest(dataset_dir):
    """Validate the manual license attestation and prepared split structure."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Dataset non vérifié : manifeste absent ({manifest_path}).")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("verified") is not True:
        raise ValueError("Le manifeste doit porter une validation manuelle explicite.")
    if manifest.get("license_confirmed_by_operator") is not True:
        raise ValueError("La confirmation manuelle de licence est absente.")
    for field in ("source", "license", "verification_evidence"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Le manifeste ne contient pas {field}.")

    classes = manifest.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or classes != sorted(set(classes))
        or any(not isinstance(name, str) or not name for name in classes)
    ):
        raise ValueError("Les classes doivent être une liste non vide, unique et triée.")

    expected_classes = set(classes)
    for split_name in SPLITS:
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            raise ValueError(f"Le découpage {split_name} est absent.")
        actual_classes = {
            entry.name for entry in split_dir.iterdir() if entry.is_dir()
        }
        if actual_classes != expected_classes:
            raise ValueError(
                f"Les classes du découpage {split_name} ne correspondent pas au manifeste."
            )
        for class_name in classes:
            class_dir = split_dir / class_name
            if not any(path.is_file() for path in class_dir.iterdir()):
                raise ValueError(f"Aucune image dans {split_name}/{class_name}.")
    return classes


def _preprocess_dataset(dataset, mobilenet_preprocess, augmentation=None, tf=None):
    """Apply the exact MobileNetV2 preprocessing; augmentation is train-only."""
    if augmentation is None:
        return dataset.map(
            lambda images, labels: (
                preprocess_mobilenet_v2(images, mobilenet_preprocess),
                labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.map(
        lambda images, labels: (
            preprocess_mobilenet_v2(
                augmentation(images, training=True), mobilenet_preprocess
            ),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _make_callbacks(tf, checkpoint_path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def _evaluate_and_write_reports(tf, model, test_dataset, class_names, report_dir):
    """Evaluate once on held-out test data and write per-class metrics and confusion plot."""
    from matplotlib import pyplot as plt

    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])
    predictions = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    confusion = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=len(class_names)
    ).numpy()
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    per_class = {}

    for index, class_name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[index, :].sum()),
        }

    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "test": {
                    "accuracy": accuracy,
                    "samples": total,
                    "per_class": per_class,
                    "confusion_matrix": confusion.tolist(),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(max(8, len(class_names) * 0.8), 7))
    axis.imshow(confusion, interpolation="nearest", cmap="Greens")
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Classe prédite",
        ylabel="Classe réelle",
        title="Matrice de confusion — jeu de test réservé",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics_path


def train_farmview_model(
    dataset_dir=DEFAULT_DATASET_DIR,
    epochs=20,
    fine_tune_epochs=10,
    batch_size=32,
    seed=2026,
):
    """Train and evaluate MobileNetV2; fail explicitly if data or runtime is unavailable."""
    if batch_size != 32:
        raise ValueError("Le pipeline FarmView utilise batch_size=32.")
    class_names = validate_dataset_manifest(dataset_dir)

    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    tf.keras.utils.set_random_seed(seed)
    options = tf.data.Options()
    options.experimental_deterministic = True
    split_datasets = {}
    for split_name in SPLITS:
        raw_dataset = tf.keras.utils.image_dataset_from_directory(
            str(Path(dataset_dir) / split_name),
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=split_name == "train",
            seed=seed,
        )
        raw_dataset = raw_dataset.with_options(options)
        split_datasets[split_name] = raw_dataset

    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ],
        name="training_only_augmentation",
    )
    train_dataset = _preprocess_dataset(
        split_datasets["train"], preprocess_input, augmentation, tf
    ).prefetch(tf.data.AUTOTUNE)
    val_dataset = _preprocess_dataset(
        split_datasets["val"], preprocess_input, tf=tf
    ).prefetch(tf.data.AUTOTUNE)
    test_dataset = _preprocess_dataset(
        split_datasets["test"], preprocess_input, tf=tf
    ).prefetch(tf.data.AUTOTUNE)

    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.25)(features)
    outputs = layers.Dense(len(class_names), activation="softmax")(features)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="farmview-training-", dir=MODEL_DIR) as temp:
        temp_dir = Path(temp)
        head_checkpoint = temp_dir / "head_best.keras"
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=_make_callbacks(tf, head_checkpoint),
            verbose=1,
        )
        best_head = tf.keras.models.load_model(head_checkpoint)
        best_head_val_loss = float(best_head.evaluate(val_dataset, verbose=0)[0])

        fine_tune_base = next(
            layer for layer in best_head.layers if isinstance(layer, tf.keras.Model)
        )
        fine_tune_base.trainable = True
        for layer in fine_tune_base.layers[:-30]:
            layer.trainable = False
        for layer in fine_tune_base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        best_head.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_checkpoint = temp_dir / "fine_tune_best.keras"
        best_head.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=fine_tune_epochs,
            callbacks=_make_callbacks(tf, fine_tune_checkpoint),
            verbose=1,
        )
        fine_tuned = tf.keras.models.load_model(fine_tune_checkpoint)
        fine_tuned_val_loss = float(fine_tuned.evaluate(val_dataset, verbose=0)[0])
        selected_model = fine_tuned if fine_tuned_val_loss <= best_head_val_loss else tf.keras.models.load_model(head_checkpoint)

        _evaluate_and_write_reports(tf, selected_model, test_dataset, class_names, REPORT_DIR)
        staged_model = temp_dir / "farmview_model.keras"
        selected_model.save(staged_model)
        staged_classes = temp_dir / "farmview_model.classes.json"
        staged_classes.write_text(
            json.dumps({"classes": class_names}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(staged_model, MODEL_DIR / "farmview_model.keras")
        os.replace(staged_classes, MODEL_DIR / "farmview_model.classes.json")

    return MODEL_DIR / "farmview_model.keras"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    model_path = train_farmview_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=32,
        seed=args.seed,
    )
    print(f"Modèle évalué et exporté : {model_path}")


if __name__ == "__main__":
    main()
