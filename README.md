# Keratoconus Classification Using Deep Learning

## Project Overview

This project explores the use of deep learning and computer vision to classify corneal images into three categories: **Keratoconus**, **Normal**, and **Suspect**.

A Convolutional Neural Network (CNN) was developed using TensorFlow and Keras to investigate how image-based machine learning can support research into automated corneal image classification.

The project demonstrates practical experience in data preparation, image preprocessing, model development, model training, and performance evaluation.

> **Important:** This is an educational and research project. The model has not been clinically validated and must not be used to diagnose or make medical decisions.

## Objectives

- Prepare and organise corneal image data for machine learning.
- Build a CNN for three-class image classification.
- Train and validate a deep learning model.
- Evaluate model performance and identify potential overfitting.
- Explore the potential applications and limitations of AI in eye health research.

## Technologies Used

- **Python** — programming and data processing
- **TensorFlow / Keras** — deep learning and CNN development
- **NumPy** — numerical operations
- **Pandas** — data handling
- **OpenCV** — image processing
- **Matplotlib** — visualisation
- **Scikit-learn** — model evaluation and classification metrics

## Dataset

The project uses corneal images organised into three classes:

- Keratoconus
- Normal
- Suspect

The training and validation dataset contains 2,388 images, while the independent test directory contains 1,050 images.

The training and validation images were divided using an 80/20 split. Images were resized to 180 × 180 pixels for the CNN input.

**Data availability:** The original dataset is stored separately from this repository. Access to the images may depend on the original dataset's source and permissions.

## Model Development

The project uses a Convolutional Neural Network to learn visual patterns from corneal images and predict one of the three classes.

The workflow includes:

1. Loading and organising image data.
2. Resizing images and preparing batches.
3. Creating the CNN architecture.
4. Training the model on the training dataset.
5. Monitoring performance on the validation dataset.
6. Identifying opportunities to improve generalisation and evaluation.

## Initial Results

The original notebook recorded a **best validation accuracy of approximately 91.19%** during training.

This is an initial result rather than a final estimate of real-world performance. The difference between training and validation performance suggests that further investigation into overfitting, data splitting, and generalisation is necessary.

Independent test-set evaluation and additional classification metrics should be used to establish a more complete picture of performance.

## Limitations and Future Improvements

Planned improvements include:

- Evaluate the model on the independent test set.
- Generate a confusion matrix and per-class precision, recall, and F1-scores.
- Investigate class imbalance and possible data leakage.
- Use early stopping and learning-rate scheduling during training.
- Compare alternative CNN architectures and transfer-learning models.
- Improve reproducibility with a documented environment and configuration.
- Investigate patient-level data splitting where patient identifiers are available.

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Obtain the dataset from its authorised source.
4. Update the dataset paths in the notebook to match your environment.
5. Run the notebook cells in sequence.

The original notebook was developed in a Google Colab environment and may require path changes to run locally.

## Skills Demonstrated

This project demonstrates experience with:

- Python programming
- Deep learning and computer vision
- Data preprocessing
- Model training and validation
- Performance analysis
- Identifying model limitations
- Communicating technical findings responsibly

## Disclaimer

This project is for educational and research purposes only. It is not a medical device, has not been clinically validated, and is not intended to replace assessment by a qualified eye-care professional.
