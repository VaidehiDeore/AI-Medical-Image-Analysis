
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import io

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "demo_model.keras"

app = Flask(__name__)

# This order matches the alphabetical class order used during training.
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = (128, 128)

model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run: python src\\train_model.py"
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


@app.get("/")
def home():
    return send_file(BASE_DIR / "index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "status": "online",
        "model_exists": MODEL_PATH.exists(),
        "notice": "Synthetic demo model only; not for medical use."
    })


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image using the 'image' field."}), 400

    uploaded = request.files["image"]

    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("L")
        image = image.resize(IMAGE_SIZE)

        # Match the training pipeline: grayscale pixels scaled to 0–1.
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=(0, -1))

        active_model = load_model()
        probabilities = active_model.predict(array, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        score = float(probabilities[predicted_index]) * 100

        return jsonify({
            "predicted_class": predicted_class,
            "class_score": round(score, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "model_type": "synthetic_demo",
            "warning": "Educational software demo only. Not a medical prediction."
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the server terminal."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
