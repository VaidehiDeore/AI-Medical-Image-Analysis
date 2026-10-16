"""Predict on one image using the synthetic demo model. Not for clinical use."""
from pathlib import Path
import sys
import tensorflow as tf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models" / "demo_model.keras"
if not model_path.exists():
    raise SystemExit("Model not found. Run: python src/train_model.py")
if len(sys.argv) < 2:
    raise SystemExit('Usage: python src/predict_image.py "path/to/image.png"')
model = tf.keras.models.load_model(model_path)
img = tf.keras.utils.load_img(sys.argv[1], color_mode="grayscale", target_size=(128,128))
arr = tf.keras.utils.img_to_array(img)[None, ...] / 255.0
scores = model.predict(arr, verbose=0)[0]
names = ["NORMAL", "PNEUMONIA"]  # alphabetic folder order in this dataset
idx = int(np.argmax(scores))
print(f"Predicted synthetic class: {names[idx]}")
print(f"Class score: {scores[idx]*100:.1f}%")
print("WARNING: synthetic demo model only; this is not a medical prediction.")
