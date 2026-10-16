# Synthetic demo dataset (software testing only)

This folder contains 120 procedurally generated, X-ray-inspired grayscale illustrations:
- train: 40 NORMAL-style + 40 PNEUMONIA-style
- val: 10 + 10
- test: 10 + 10

**Critical limitation:** These are artificial drawings with stylized visual differences, NOT real medical images and NOT medically valid labels. They are supplied only to verify that file loading, preprocessing, training code, evaluation, and dashboard plumbing execute end-to-end. A model trained on these files must not be described as detecting pneumonia and its metrics must not be reported as medical AI performance.

For genuine experiments, replace this folder with a properly licensed public dataset such as a public chest X-ray dataset, review its license and documentation, and preserve a patient-independent held-out test set where metadata allows.
