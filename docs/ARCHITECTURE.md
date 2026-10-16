# Architecture

```text
Public labeled images
        |
        v
Dataset audit + split (train / validation / test)
        |
        v
Preprocessing (decode, resize, scale; augmentation on train only)
        |
        v
CNN / transfer-learning model
        |
        +--> Training history + saved model
        |
        v
Held-out evaluation (confusion matrix, sensitivity, specificity, F1, ROC-AUC)
        |
        v
Inference adapter (same preprocessing as training)
        |
        v
Dashboard result presentation + limitations notice
```

The current `index.html` implements only the frontend interaction and a simulated result. A real model integration must replace the simulation with actual inference from a trained model, and must clearly report the model/dataset version.
