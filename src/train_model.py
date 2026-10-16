"""Train a small CNN on the included synthetic demo dataset.
WARNING: Synthetic data is for pipeline smoke-testing only, not medical inference."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_demo"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
IMG_SIZE = (128, 128)
BATCH = 8
SEED = 42

def loader(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA / split, image_size=IMG_SIZE, color_mode="grayscale",
        batch_size=BATCH, shuffle=shuffle, seed=SEED
    )

train_ds = loader("train", True)
val_ds = loader("val", False)
test_ds = loader("test", False)
class_names = train_ds.class_names
normalizer = tf.keras.layers.Rescaling(1./255)

def prep(ds):
    return ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)

train_ds, val_ds, test_ds = prep(train_ds), prep(val_ds), prep(test_ds)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128,128,1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODELS / "demo_model.keras", save_best_only=True, monitor="val_loss")
]
history = model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)
model.save(MODELS / "demo_model.keras")

# Curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy (synthetic data)"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss (synthetic data)"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=150); plt.close(fig)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probs, axis=1)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix — Synthetic Demo Only")
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
with open(OUT / "metrics_synthetic_demo.json", "w", encoding="utf-8") as f:
    json.dump({"warning": "Synthetic demo only; not medical performance.", "class_names": class_names, "report": report}, f, indent=2)
print("Finished. Outputs saved in outputs/. These metrics are only for synthetic pipeline testing.")
