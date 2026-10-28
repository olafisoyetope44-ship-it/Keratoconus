# Keratoconus  Classification with Deep Learning

A TensorFlow/Keras computer-vision project exploring three-class classification of corneal images: **Keratoconus**, **Normal**, and **Suspect**.

> **Research/portfolio project only.** This model is not a medical device, has not been clinically validated, and must not be used to diagnose, screen, or guide treatment. Predictions require evaluation on appropriate, independent clinical data before any real-world interpretation.

## Project overview

This project uses a convolutional neural network (CNN) to learn image features and classify images into three categories. The original notebook is `Keratoconus_Deep_Learning_Final.ipynb`.

### Current notebook findings

The notebook's saved outputs show:
- 2,388 images in the training/validation directory, split into 1,911 training images and 477 validation images.
- 1,050 images detected in the separate independent test directory.
- Best saved validation accuracy of approximately **91.19%** during the original run.
- Training accuracy rose to approximately **99.58%**, while validation accuracy later declined, which is a sign of possible overfitting.

**Important:** the 91.19% figure is validation accuracy, not independent-test accuracy. The original notebook does not evaluate its independent test set, so no test-set performance is claimed here. The original model also defines five output units despite having three dataset classes; retraining with the corrected three-class setup is required before reporting new results.

## Repository contents

- `Keratoconus_Deep_Learning_Final.ipynb` — original exploratory notebook.
- `model_evaluation.py` — reproducible training/evaluation workflow with a three-class output layer, callbacks, independent test evaluation, and per-class metrics.

## Dataset layout

Keep the dataset outside version control unless you have permission to redistribute it. Arrange local files like this:

```text
Keratoconus/
├── Keratoconus_Deep_Learning_Final.ipynb
├── model_evaluation.py
└── data/
    ├── Train_Validation sets/
    │   ├── Keratoconus/
    │   ├── Normal/
    │   └── Suspect/
    └── Independent Test Set/
        ├── Keratoconus/
        ├── Normal/
        └── Suspect/
```

Folder names are used as class labels. The training and test class names must match exactly.

## Run locally

Use Python 3.10 or 3.11 in a virtual environment, then install:

```bash
pip install tensorflow scikit-learn matplotlib numpy
```

Set the dataset root to the folder containing both dataset directories:

**Windows PowerShell**
```powershell
$env:KERATOCONUS_DATA_ROOT = "C:\path\to\data"
python model_evaluation.py
```

**macOS/Linux**
```bash
export KERATOCONUS_DATA_ROOT="/path/to/data"
python model_evaluation.py
```

The script saves the best validation model to `artifacts/keratoconus_cnn.keras`, training curves to `artifacts/training_curves.png`, a confusion matrix to `artifacts/confusion_matrix.png`, and a per-class classification report to `artifacts/classification_report.txt`.

## Evaluation and limitations

The evaluation workflow reports independent test accuracy, a confusion matrix, and per-class precision, recall, and F1-score. Do not use the test set to tune hyperparameters; keep it as a final holdout.

A random image-level validation split can overestimate performance if images from the same patient appear in both training and validation sets. If patient identifiers are available, use patient-level grouping for the split and document the method. Also report class counts and investigate class imbalance before drawing conclusions.

## Next improvements

- Verify patient-level separation and dataset provenance.
- Compare a small CNN with transfer learning and a simple baseline.
- Add class-imbalance checks and suitable augmentation.
- Record exact package versions, split strategy, and random seeds.
- Add an honest results table after the corrected pipeline has been run.
