import os
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def collect_video_paths(dataset_dir: str) -> Tuple[List[str], List[int], List[str]]:
    dataset_root = Path(dataset_dir)
    real_dir = dataset_root / "real"
    fake_dir = dataset_root / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(f"Dataset directories not found. Expected {real_dir} and {fake_dir}.")
    video_paths = []
    labels = []
    video_names = []
    for label_dir, label in [(real_dir, 1), (fake_dir, 0)]:
        if not label_dir.exists():
            continue
        for video_path in sorted(label_dir.rglob("*")):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                video_paths.append(str(video_path))
                labels.append(label)
                video_names.append(video_path.name)
    if not video_paths:
        raise ValueError(f"No videos found in {dataset_root}. Please organize dataset as dataset/real/ and dataset/fake/")
    print(f"Found {len(video_paths)} total videos")
    print(f"  - Real videos: {sum(1 for l in labels if l == 1)}")
    print(f"  - Fake videos: {sum(1 for l in labels if l == 0)}")
    return video_paths, labels, video_names

def split_videos_by_video(video_paths: List[str], labels: List[int], train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42):
    from sklearn.model_selection import train_test_split
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    video_paths = np.asarray(video_paths)
    labels = np.asarray(labels)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(video_paths, labels, train_size=train_ratio, stratify=labels, random_state=random_state)
    remaining_ratio = val_ratio + test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, train_size=(val_ratio / remaining_ratio), stratify=temp_labels, random_state=random_state)
    return {"train": (train_paths.tolist(), train_labels.tolist()), "val": (val_paths.tolist(), val_labels.tolist()), "test": (test_paths.tolist(), test_labels.tolist())}

def sample_video_frames(video_path: str, frames_per_video: int = 60, target_size: tuple = (224, 224), channels: int = 3) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")
    frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Could not extract any frames from video: {video_path}")
    while len(frames) < frames_per_video:
        frames.append(frames[-1].copy())
    sequence = np.stack(frames[:frames_per_video], axis=0)
    return sequence.astype(np.float32)

def create_sequence_dataset(video_paths: List[str], labels: List[int], frames_per_video: int = 60, batch_size: int = 8, shuffle: bool = True, seed: int = 42):
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    indices = np.arange(len(video_paths))
    if shuffle:
        rng.shuffle(indices)
    def generator():
        for idx in indices:
            video_path = video_paths[int(idx)]
            label = float(labels[int(idx)])
            try:
                sequence = sample_video_frames(video_path, frames_per_video=frames_per_video)
                yield sequence, label
            except Exception as e:
                print(f"Warning: Could not process {video_path}: {e}")
                continue
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(frames_per_video, 224, 224, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(32, len(video_paths)))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
