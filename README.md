
# AI-Powered Medical Image Analysis System

A Python-based medical image analysis prototype that demonstrates image preprocessing, CNN model training, image classification, and result visualization through a responsive MEDORA healthcare dashboard.

> **Disclaimer:** This is an educational software prototype trained on synthetic X-ray-inspired illustrations. It is not trained or validated for medical diagnosis and must not be used for clinical decisions.

## Features

- Responsive MEDORA dashboard with image upload and preview.
- Flask API for connecting the web interface to a trained TensorFlow/Keras model.
- Image preprocessing using grayscale conversion, resizing, and pixel normalization.
- CNN-based binary classification workflow with `NORMAL` and `PNEUMONIA` demonstration classes.
- Training curves, confusion matrix, and evaluation metrics.
- Prediction results with class scores and a clear non-clinical disclaimer.
- Basic automated tests for model and output files.

## Technology Stack

- **Language:** Python
- **Machine Learning:** TensorFlow, Keras
- **Image Processing:** Pillow, NumPy
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Evaluation:** Matplotlib, scikit-learn, pytest

## Project Structure

```text
AI-Medical-Image-Analysis/
├── data/
│   └── synthetic_demo/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── LIMITATIONS.md
│   └── SEVEN_DAY_PROOF_PLAN.md
├── models/
│   └── demo_model.keras
├── outputs/
│   ├── confusion_matrix.png
│   ├── metrics_synthetic_demo.json
│   └── training_curves.png
├── screenshots/
├── src/
│   ├── predict_image.py
│   └── train_model.py
├── tests/
│   └── test_project.py
├── app.py
├── index.html
├── requirements.txt
└── .gitignore
```

## Installation and Setup

**Requirements:** Python 3.11 is recommended for the current project environment.

1. Clone the repository and open its directory:

   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Medical-Image-Analysis.git
   cd AI-Medical-Image-Analysis
   ```

2. Create and activate a virtual environment.

   **Windows PowerShell:**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install pytest
   ```

## Running the Project

### 1. Train the demonstration model

Run this command from the project root:

```bash
python src/train_model.py
```

The script trains the CNN using the synthetic dataset and generates the model and evaluation artifacts in `models/` and `outputs/`.

### 2. Run a prediction from the terminal

```bash
python src/predict_image.py "data/synthetic_demo/test/NORMAL/synthetic_normal_001.png"
```

The output includes a predicted demonstration class and class score.

### 3. Launch the MEDORA dashboard

Start the Flask server:

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

Upload an image or use the demo-image option, then run the model demo to view the returned class and score.

### 4. Run automated tests

```bash
python -m pytest -q
```

The current tests check that the trained model, evaluation files, and metrics JSON are present and readable. They do not measure medical accuracy.

## Model Evaluation

Training curves, a confusion matrix, and metrics are saved in the `outputs/` directory. The current dataset is synthetic and intended only to verify that the training and inference pipeline runs.

Validation performance on synthetic illustrations does not establish generalization to real medical images.

## Limitations and Responsible Use

- The bundled images are synthetic illustrations, not real chest X-rays.
- The `NORMAL` and `PNEUMONIA` labels are demonstration classes, not clinically validated predictions.
- Class scores are model outputs, not calibrated medical confidence.
- The project is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.
- Do not upload identifiable patient information.
- Any future experiment using real medical images must use an appropriately licensed dataset, documented data splits, suitable evaluation, and a thorough review of privacy and model limitations.

## Future Improvements

- Experiment with an appropriately licensed real medical imaging dataset.
- Evaluate model performance using a separate, properly labeled test set.
- Add stronger input validation, error handling, and model evaluation tests.
- Improve reproducibility and document dataset provenance and limitations.

## License

Choose and add a suitable open-source license before distributing the project. Ensure that any future dataset and third-party assets are used according to their respective licenses.
