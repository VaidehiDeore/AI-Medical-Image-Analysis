
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_trained_model_exists():
    assert (ROOT / "models" / "demo_model.keras").is_file()


def test_training_outputs_exist():
    assert (ROOT / "outputs" / "training_curves.png").is_file()
    assert (ROOT / "outputs" / "confusion_matrix.png").is_file()


def test_synthetic_metrics_file_is_valid_json():
    metrics_file = ROOT / "outputs" / "metrics_synthetic_demo.json"
    assert metrics_file.is_file()

    with metrics_file.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    assert isinstance(metrics, dict)
