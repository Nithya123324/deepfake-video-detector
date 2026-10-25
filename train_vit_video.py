
import random
import re
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ================= CONFIGURATION =================

SEED = 42

FRAMES_PER_VIDEO = 8
IMAGE_SIZE = 64
PATCH_SIZE = 8

BATCH_SIZE = 4
EPOCHS = 25

MODEL_PATH = "vit_video_model.keras"

REAL_DIR = Path("dataset/real")
FAKE_DIR = Path("dataset/fake")

# =================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def get_videos():
    """Collect video paths and labels. FAKE=0, REAL=1."""

    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    records = []

    for folder, label in [
        (FAKE_DIR, 0),
        (REAL_DIR, 1),
    ]:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                records.append((str(path), label))

    if not records:
        raise ValueError("No video files found.")

    return records


def get_group_id(path):
    """
    Group augmented versions of the same original video.
    """

    name = Path(path).stem

    # Remove augmentation number and any following random suffix.
    name = re.sub(r"_aug_\d+.*$", "", name)

    return name


def load_video(path):
    """Extract uniformly sampled frames from one video."""

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No readable frames: {path}")

    indices = np.linspace(
        0,
        total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int,
    )

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

        success, frame = cap.read()

        if success and frame is not None:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frame = cv2.resize(
                frame,
                (IMAGE_SIZE, IMAGE_SIZE),
            )

            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"Frame extraction failed: {path}")

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(frames[-1].copy())

    return np.stack(frames[:FRAMES_PER_VIDEO])


class VideoSequence(tf.keras.utils.Sequence):
    """Load video batches without loading the entire dataset into RAM."""

    def __init__(
        self,
        records,
        batch_size=BATCH_SIZE,
        shuffle=False,
    ):
        super().__init__()

        self.records = list(records)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.order = np.arange(len(self.records))

        self.on_epoch_end()

    def __len__(self):
        return int(
            np.ceil(len(self.records) / self.batch_size)
        )

    def __getitem__(self, index):

        batch_ids = self.order[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        videos = []
        labels = []

        for record_id in batch_ids:

            path, label = self.records[record_id]

            videos.append(load_video(path))
            labels.append(label)

        return (
            np.asarray(videos, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
        )

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.order)


def build_vit_video_model():
    """Build Vision Transformer + BiLSTM video classifier."""

    video_input = tf.keras.Input(
        shape=(
            FRAMES_PER_VIDEO,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        name="video_frames",
    )

    # Patch embedding.
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=PATCH_SIZE,
            strides=PATCH_SIZE,
            padding="valid",
        ),
        name="patch_embedding",
    )(video_input)

    # Convert patches into tokens.
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Reshape((64, 64)),
        name="flatten_patches",
    )(x)

    # Vision Transformer encoder.
    position_input = tf.keras.Input(
        shape=(64, 64),
        name="patch_tokens",
    )

    positions = tf.keras.layers.Embedding(
        input_dim=64,
        output_dim=64,
        name="position_embedding",
    )(
        tf.range(
            start=0,
            limit=64,
            delta=1,
        )
    )

    x_frame = position_input + positions

    # Transformer block 1.
    residual = x_frame

    z = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
    )(x_frame)

    z = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=16,
        dropout=0.1,
    )(z, z)

    x_frame = tf.keras.layers.Add()([
        residual,
        z,
    ])

    # Transformer block 2.
    residual = x_frame

    z = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
    )(x_frame)

    z = tf.keras.layers.Dense(
        128,
        activation="gelu",
    )(z)

    z = tf.keras.layers.Dropout(0.1)(z)

    z = tf.keras.layers.Dense(64)(z)

    x_frame = tf.keras.layers.Add()([
        residual,
        z,
    ])

    x_frame = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
    )(x_frame)

    x_frame = tf.keras.layers.GlobalAveragePooling1D()(
        x_frame
    )

    frame_encoder = tf.keras.Model(
        position_input,
        x_frame,
        name="vit_encoder",
    )

    # Apply ViT to every frame.
    x = tf.keras.layers.TimeDistributed(
        frame_encoder,
        name="frame_vit",
    )(x)

    # BiLSTM learns temporal relationships.
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32,
            return_sequences=True,
        ),
        name="bilstm",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling1D(
        name="video_pooling",
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="gelu",
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="real_probability",
    )(x)

    model = tf.keras.Model(
        video_input,
        output,
        name="vit_bilstm_video_detector",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
            ),
            tf.keras.metrics.AUC(
                name="roc_auc",
            ),
        ],
    )

    return model


def split_dataset(records, labels, groups):
    """
    Split into training, validation, and testing sets.
    Related augmented videos remain in the same split.
    """

    # First split: 80% development, 20% testing.
    first_split = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    train_val_idx, test_idx = next(
        first_split.split(
            records,
            labels,
            groups,
        )
    )

    train_val_records = [
        records[i] for i in train_val_idx
    ]

    train_val_labels = labels[train_val_idx]

    train_val_groups = groups[train_val_idx]

    # Second split: 75% training, 25% validation
    # within the development set.
    second_split = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        second_split.split(
            train_val_records,
            train_val_labels,
            train_val_groups,
        )
    )

    train_records = [
        train_val_records[i] for i in train_idx
    ]

    val_records = [
        train_val_records[i] for i in val_idx
    ]

    test_records = [
        records[i] for i in test_idx
    ]

    return train_records, val_records, test_records


def main():

    records = get_videos()

    labels = np.asarray(
        [label for _, label in records],
        dtype=np.int32,
    )

    groups = np.asarray([
        get_group_id(path)
        for path, label in records
    ])

    print("\n========== DATASET ==========")

    print("Total videos:", len(records))

    print("FAKE videos:", int(np.sum(labels == 0)))

    print("REAL videos:", int(np.sum(labels == 1)))

    print("Unique video groups:", len(np.unique(groups)))

    # Group-aware dataset split.
    train_records, val_records, test_records = split_dataset(
        records,
        labels,
        groups,
    )

    print("\n========== SPLIT ==========")

    print("Training:", len(train_records))

    print("Validation:", len(val_records))

    print("Testing:", len(test_records))

    # Calculate class weights.
    train_labels = np.asarray([
        label for _, label in train_records
    ])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )

    class_weight_dict = {
        0: float(class_weights[0]),
        1: float(class_weights[1]),
    }

    print("\nClass weights:", class_weight_dict)

    # Data loaders.
    train_data = VideoSequence(
        train_records,
        shuffle=True,
    )

    val_data = VideoSequence(
        val_records,
        shuffle=False,
    )

    test_data = VideoSequence(
        test_records,
        shuffle=False,
    )

    # Build model.
    model = build_vit_video_model()

    model.summary()

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n========== TRAINING ==========")

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Load the best model.
    model = tf.keras.models.load_model(
        MODEL_PATH,
    )

    print("\n========== TESTING ==========")

    probabilities = model.predict(
        test_data,
        verbose=1,
    ).reshape(-1)

    y_true = np.asarray([
        label for _, label in test_records
    ], dtype=np.int32)

    y_pred = (
        probabilities >= 0.5
    ).astype(np.int32)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    print("\nVIDEO-LEVEL TEST RESULTS")

    print("------------------------")

    print("Accuracy:", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall:", round(recall, 4))

    print("F1-score:", round(f1, 4))

    if len(np.unique(y_true)) == 2:

        auc = roc_auc_score(
            y_true,
            probabilities,
        )

        print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0,
        )
    )

    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()
