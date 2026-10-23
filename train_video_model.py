"""Train a video-level deepfake detection model.

This script:
1. Collects videos from dataset/real/ and dataset/fake/
2. Splits them by VIDEO (not by frame) into train/val/test
3. Samples fixed number of frames from each video
4. Trains the EfficientNetB0 + ViT model
5. Evaluates on completely unseen test videos
6. Saves metrics and predictions

IMPORTANT: No frame from the same video appears in multiple splits.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset_utils import collect_video_paths, create_sequence_dataset, split_videos_by_video
from video_model import build_video_deepfake_model, compile_model

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_DIR = "dataset"
FRAMES_PER_VIDEO = 60  # Number of frames to sample from each video
BATCH_SIZE = 8  # Batch size for training
EPOCHS = 30  # Maximum number of epochs
LEARNING_RATE = 1e-4  # Adam learning rate
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.15  # 15% for validation
TEST_RATIO = 0.15  # 15% for testing (completely unseen)
RANDOM_STATE = 42  # Seed for reproducibility
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)


def save_confusion_matrix(y_true, y_pred, save_path: Path):
    """Save confusion matrix as a PNG image."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Video Level Predictions")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["REAL", "FAKE"])
    plt.yticks([0, 1], ["REAL", "FAKE"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_predictions_csv(video_paths, y_true, y_pred, y_prob, save_path: Path):
    """Save per-video predictions to CSV.
    
    Format:
    video_name, actual_label, predicted_label, confidence
    """
    df = pd.DataFrame(
        {
            "video_name": [Path(p).name for p in video_paths],
            "actual_label": ["REAL" if int(v) == 1 else "FAKE" for v in y_true],
            "predicted_label": ["REAL" if int(v) == 1 else "FAKE" for v in y_pred],
            "confidence": [float(max(p, 1.0 - p)) for p in y_prob],
        }
    )
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")
    print("\nSample predictions:")
    print(df.head(10))
    return df


def evaluate_model(model, test_dataset, test_video_paths, test_labels):
    """Evaluate model on test set and save metrics.
    
    Args:
        model: Trained Keras model
        test_dataset: tf.data.Dataset with test samples
        test_video_paths: List of test video paths
        test_labels: True labels for test videos
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING ON UNSEEN TEST VIDEOS")
    print("="*60)
    
    # Get predictions on test set
    probabilities = model.predict(test_dataset, verbose=1).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    # Collect true labels from test dataset
    true_labels = []
    for _, y in test_dataset.unbatch().batch(1):
        true_labels.extend(y.numpy().astype(int).tolist())

    # Ensure lengths match
    if len(true_labels) != len(predictions):
        print(f"Warning: Length mismatch. Trimming to {len(predictions)}")
        true_labels = true_labels[: len(predictions)]

    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision": float(precision_score(true_labels, predictions, zero_division=0)),
        "recall": float(recall_score(true_labels, predictions, zero_division=0)),
        "f1": float(f1_score(true_labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_labels, probabilities)),
    }

    # Save visualizations and data
    save_confusion_matrix(true_labels, predictions, MODEL_DIR / "confusion_matrix.png")
    save_predictions_csv(
        test_video_paths,
        true_labels,
        predictions,
        probabilities,
        MODEL_DIR / "test_predictions.csv",
    )

    # Save metrics report
    with open(MODEL_DIR / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("TEST SET EVALUATION METRICS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key:15s}: {value:.4f}")
    print("="*60)
    print(f"\nMetrics report saved to {MODEL_DIR / 'metrics_report.json'}")

    return metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train a video-level deepfake detection model."
    )
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--frames_per_video", type=int, default=FRAMES_PER_VIDEO)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    args = parser.parse_args()

    # ========================================================================
    # STEP 1: Collect video paths
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: COLLECTING VIDEOS FROM DATASET")
    print("="*60)
    
    video_paths, labels, video_names = collect_video_paths(args.dataset_dir)

    # ========================================================================
    # STEP 2: Split by VIDEO (NOT by frame)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: SPLITTING VIDEOS (VIDEO-LEVEL SPLIT)")
    print("="*60)
    
    split_data = split_videos_by_video(
        video_paths,
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=RANDOM_STATE,
    )

    train_paths, train_labels = split_data["train"]
    val_paths, val_labels = split_data["val"]
    test_paths, test_labels = split_data["test"]

    print(f"Train videos: {len(train_paths)} ({len(train_paths)/len(video_paths)*100:.1f}%)")
    print(f"  - Real: {sum(1 for l in train_labels if l == 1)}")
    print(f"  - Fake: {sum(1 for l in train_labels if l == 0)}")
    
    print(f"\nValidation videos: {len(val_paths)} ({len(val_paths)/len(video_paths)*100:.1f}%)")
    print(f"  - Real: {sum(1 for l in val_labels if l == 1)}")
    print(f"  - Fake: {sum(1 for l in val_labels if l == 0)}")
    
    print(f"\nTest videos: {len(test_paths)} ({len(test_paths)/len(video_paths)*100:.1f}%)")
    print(f"  - Real: {sum(1 for l in test_labels if l == 1)}")
    print(f"  - Fake: {sum(1 for l in test_labels if l == 0)}")

    # ========================================================================
    # STEP 3: Create tf.data datasets
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: CREATING DATASETS")
    print(f"Sampling {args.frames_per_video} frames per video")
    print("="*60)
    
    train_dataset = create_sequence_dataset(
        train_paths,
        train_labels,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataset = create_sequence_dataset(
        val_paths,
        val_labels,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_dataset = create_sequence_dataset(
        test_paths,
        test_labels,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ========================================================================
    # STEP 4: Build model
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 4: BUILDING MODEL")
    print("Architecture: EfficientNetB0 + Temporal Attention")
    print("="*60)
    
    model = build_video_deepfake_model(frames_per_video=args.frames_per_video)
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    # ========================================================================
    # STEP 5: Setup callbacks
    # ========================================================================
    checkpoint_path = MODEL_DIR / "best_video_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ========================================================================
    # STEP 6: Train model
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 5: TRAINING MODEL")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best model saved to: {checkpoint_path}")
    print("="*60)

    # ========================================================================
    # STEP 7: Evaluate on test set
    # ========================================================================
    # Load best model
    model = tf.keras.models.load_model(str(checkpoint_path), compile=False)
    model = compile_model(model, learning_rate=args.learning_rate)
    
    evaluate_model(model, test_dataset, test_paths, test_labels)

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - Model: {checkpoint_path}")
    print(f"  - Metrics: {MODEL_DIR / 'metrics_report.json'}")
    print(f"  - Predictions: {MODEL_DIR / 'test_predictions.csv'}")
    print(f"  - Confusion matrix: {MODEL_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
