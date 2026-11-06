import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"
MINIMUM_TEST_ACCURACY = 0.60


@pytest.mark.skipif(not METRICS_PATH.is_file(), reason="Aucune métrique de test réelle n'est disponible.")
def test_held_out_accuracy_exceeds_six_class_baseline():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert metrics["test"]["accuracy"] >= MINIMUM_TEST_ACCURACY, (
        "Seuil de CI fixé à 60% : environ 3,6 fois le hasard uniforme à 1/6; "
        "ce seuil de fumée ne constitue pas une qualification agronomique."
    )
