
import os
import logging
import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """Video detector using Vision Transformer and BiLSTM."""

    
    def __init__(self):
        self.model = None
        self.frames_per_video = 8
        self.image_size = 64
        self.model_loaded = False

        self.load_model()

    def load_model(self):
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "vit_video_model.keras"
            )

            if not os.path.isfile(model_path):
                logger.error("vit_video_model.keras not found.")
                return False

            self.model = tf.keras.models.load_model(model_path)

            if tuple(self.model.input_shape[1:]) != (8, 64, 64, 3):
                logger.error("Unexpected ViT input shape: %s",
                             self.model.input_shape)
                self.model = None
                return False

            self.model_loaded = True
            logger.info("ViT + BiLSTM model loaded: %s",
                        self.model.input_shape)
            return True

        except Exception:
            logger.exception("Could not load ViT model.")
            self.model = None
            self.model_loaded = False
            return False

    def predict_video(self, video_path):
        if not self.model_loaded or self.model is None:
            return None, "ViT model not loaded"

        cap = cv2.VideoCapture(video_path)

        try:
            if not cap.isOpened():
                return None, "Could not open video"

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))

            if total_frames <= 0:
                return None, "Video contains no readable frames"

            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0

            duration = total_frames / fps
            indices = np.linspace(
                0, total_frames - 1,
                self.frames_per_video, dtype=int
            )

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()

                if ok and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(
                        frame, (self.image_size, self.image_size)
                    )
                    frames.append(frame.astype(np.float32) / 255.0)

            if not frames:
                return None, "Could not extract video frames"

            while len(frames) < self.frames_per_video:
                frames.append(frames[-1].copy())

            video_input = np.expand_dims(
                np.stack(frames[:self.frames_per_video]), axis=0
            )

            score = float(
                np.asarray(
                    self.model.predict(video_input, verbose=0)
                ).reshape(-1)[0]
            )

            if not np.isfinite(score):
                return None, "Model returned an invalid prediction"

            real_probability = float(np.clip(score, 0, 1))
            fake_probability = 1.0 - real_probability

            prediction = "REAL" if real_probability >= 0.5 else "FAKE"
            confidence = max(real_probability, fake_probability)

            return {
                "prediction": prediction,
                "confidence": confidence,
                "real_probability": real_probability,
                "fake_probability": fake_probability,
                "frames_analyzed": self.frames_per_video,
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "selected_frames": self.frames_per_video
            }, None

        except Exception as exc:
            logger.exception("Video prediction failed.")
            return None, str(exc)

        finally:
            cap.release()
