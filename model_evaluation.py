"""Train and evaluate a three-class Keratoconus image classifier.

Expected data layout:
  <data-root>/Train_Validation sets/<class-name>/*
  <data-root>/Independent Test Set/<class-name>/*

Set KERATOCONUS_DATA_ROOT to the directory containing those two folders.
This is an educational/research example, not a clinically validated tool.
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

SEED = 123
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 30
DATA_ROOT = Path(os.environ.get("KERATOCONUS_DATA_ROOT", "data"))
TRAIN_VALIDATION_DIR = DATA_ROOT / "Train_Validation sets"
TEST_DIR = DATA_ROOT / "Independent Test Set"
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


def load_datasets():
    """Load RGB images with identical image size and class ordering."""
    if not TRAIN_VALIDATION_DIR.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_VALIDATION_DIR.resolve()}\n"
            "Set KERATOCONUS_DATA_ROOT to the directory containing the dataset folders."
        )
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Independent test folder not found: {TEST_DIR.resolve()}"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_VALIDATION_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="int",
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_VALIDATION_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="int",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )

    if len(class_names) != 3:
        raise ValueError(
            f"Expected exactly 3 classes, found {len(class_names)}: {class_names}"
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a compact CNN with output units matching the dataset classes."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)  # logits
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("Classes:", class_names)

    model = build_model(num_classes=len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "keratoconus_cnn.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Plot the actual number of epochs completed.
    epochs_ran = range(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_ran, history.history["accuracy"], label="Training")
    axes[0].plot(epochs_ran, history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(epochs_ran, history.history["loss"], label="Training")
    axes[1].plot(epochs_ran, history.history["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    # Evaluate only after model development is complete.
    probabilities = tf.nn.softmax(model.predict(test_ds, verbose=0), axis=1).numpy()
    y_pred = probabilities.argmax(axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"Independent test accuracy: {test_accuracy:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    print(report)
    (OUTPUT_DIR / "classification_report.txt").write_text(
        f"Independent test accuracy: {test_accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Independent Test Set — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")
    print(
        "Reminder: results are not clinical validation. Check patient-level "
        "separation and dataset provenance before interpreting performance."
    )


if __name__ == "__main__":
    main()
