#!/usr/bin/env python3

"""
Deepfake Video Detector Web Interface
Flask application for deepfake video detection
"""

import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import cv2
from werkzeug.utils import secure_filename
import logging


# ======================================================
# CONFIGURE LOGGING
# ======================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================
# INITIALIZE FLASK APP
# ======================================================

app = Flask(__name__)

# Maximum upload size: 500 MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'deepfake_video_detector_secret_key_2024'

os.makedirs(
    app.config['UPLOAD_FOLDER'],
    exist_ok=True
)


# ======================================================
# DEEPFAKE VIDEO DETECTOR
# ======================================================

class DeepfakeDetector:
    """Deepfake video detection class"""

    def __init__(self):

        self.model = None

        # Model input size
        self.input_shape = (224, 224, 3)

        self.model_loaded = False


    # ==================================================
    # LOAD MODEL
    # ==================================================

    def load_model(self):
        """Load the trained deepfake detection model"""

        try:

            model_paths = [
                'deepfake_detector_savedmodel',
                'hybrid_deepfake_detector_savedmodel'
            ]

            for model_path in model_paths:

                if os.path.exists(model_path):

                    logger.info(
                        f"Loading model from {model_path}"
                    )

                    loaded_model = tf.saved_model.load(
                        model_path
                    )

                    if hasattr(
                        loaded_model,
                        'signatures'
                    ):

                        if 'serving_default' in loaded_model.signatures:

                            inference_func = (
                                loaded_model.signatures[
                                    'serving_default'
                                ]
                            )

                            def predict_wrapper(x):

                                if not isinstance(
                                    x,
                                    tf.Tensor
                                ):

                                    x = tf.convert_to_tensor(
                                        x,
                                        dtype=tf.float32
                                    )

                                input_key = 'input_1'
                                output_key = 'dense_2'

                                result = inference_func(
                                    **{
                                        input_key: x
                                    }
                                )

                                return result[
                                    output_key
                                ].numpy()

                            self.model = predict_wrapper
                            self.model_loaded = True

                            logger.info(
                                "Model loaded successfully!"
                            )

                            return True

            logger.error(
                "No compatible model found!"
            )

            return False

        except Exception as e:

            logger.error(
                f"Model loading failed: {e}"
            )

            return False


    # ==================================================
    # FRAME PREPROCESSING
    # ==================================================

    def preprocess_frame(self, frame):
        """
        Preprocess one video frame
        for model prediction.
        """

        try:

            # Resize frame
            frame = cv2.resize(
                frame,
                (
                    self.input_shape[0],
                    self.input_shape[1]
                )
            )

            # Convert BGR to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Normalize pixel values
            frame = frame.astype(
                np.float32
            ) / 255.0

            # Add batch dimension
            frame = np.expand_dims(
                frame,
                axis=0
            )

            return frame

        except Exception as e:

            logger.error(
                f"Frame preprocessing failed: {e}"
            )

            return None


    # ==================================================
    # SINGLE FRAME PREDICTION
    # ==================================================

    def predict_frame(self, frame):
        """
        Predict whether a video frame is REAL or FAKE.
        """

        try:

            if not self.model_loaded:

                return None, "Model not loaded"

            processed_frame = (
                self.preprocess_frame(frame)
            )

            if processed_frame is None:

                return None, (
                    "Frame preprocessing failed"
                )

            prediction = self.model(
                processed_frame
            )

            pred_value = float(
                np.asarray(
                    prediction
                ).flatten()[0]
            )

            # Classification
            if pred_value > 0.5:

                predicted_class = "FAKE"
                confidence = pred_value

            else:

                predicted_class = "REAL"
                confidence = 1 - pred_value

            result = {

                'prediction':
                    predicted_class,

                'confidence':
                    float(confidence),

                'fake_probability':
                    float(pred_value),

                'real_probability':
                    float(1 - pred_value)
            }

            return result, None

        except Exception as e:

            logger.error(
                f"Frame prediction failed: {e}"
            )

            return None, str(e)


    # ==================================================
    # VIDEO PREDICTION
    # ==================================================

    def predict_video(self, video_path):
        """
        Predict whether a video is REAL or FAKE.

        The number of analyzed frames automatically
        increases according to the video duration.
        """

        try:

            if not self.model_loaded:
                return None, "Model not loaded"

            # ------------------------------------------
            # OPEN VIDEO
            # ------------------------------------------

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return None, "Could not open video"

            # ------------------------------------------
            # VIDEO INFORMATION
            # ------------------------------------------

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30

            duration = total_frames / fps

            if total_frames <= 0:

                cap.release()

                return None, "Video contains no frames"

            logger.info(
                f"Total frames: {total_frames}"
            )

            logger.info(
                f"FPS: {fps:.2f}"
            )

            logger.info(
                f"Duration: {duration:.2f} seconds"
            )

            # ------------------------------------------
            # AUTOMATIC FRAME COUNT
            # ------------------------------------------

            if duration <= 60:
                max_frames = 30

            elif duration <= 5 * 60:
                max_frames = 60

            elif duration <= 10 * 60:
                max_frames = 120

            elif duration <= 20 * 60:
                max_frames = 240

            else:
                max_frames = 360

            # ------------------------------------------
            # SELECT FRAMES EVENLY
            # ------------------------------------------

            frame_indices = np.linspace(
                0,
                total_frames - 1,
                number_of_frames,
                dtype=int
            )

            fake_probabilities = []

            successful_frames = 0

            # ------------------------------------------
            # PROCESS FRAMES
            # ------------------------------------------

            for frame_number, frame_index in enumerate(
                frame_indices,
                start=1
            ):

                logger.info(
                    f"Processing frame "
                    f"{frame_number}/{number_of_frames}"
                )

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index)
                )

                success, frame = cap.read()

                if not success:

                    logger.warning(
                        f"Could not read frame "
                        f"{frame_index}"
                    )

                    continue

                result, error = self.predict_frame(
                    frame
                )

                if error:

                    logger.warning(
                        f"Frame prediction failed: "
                        f"{error}"
                    )

                    continue

                fake_probabilities.append(
                    result['fake_probability']
                )

                successful_frames += 1

            # ------------------------------------------
            # RELEASE VIDEO
            # ------------------------------------------

            cap.release()

            # ------------------------------------------
            # CHECK RESULTS
            # ------------------------------------------

            if not fake_probabilities:

                return None, (
                    "Could not process any video frames"
                )

            # ------------------------------------------
            # AVERAGE PREDICTION
            # ------------------------------------------

            average_fake_probability = float(
                np.mean(fake_probabilities)
            )

            average_real_probability = (
                1 - average_fake_probability
            )

            # ------------------------------------------
            # FINAL CLASSIFICATION
            # ------------------------------------------

            if average_fake_probability > 0.5:

                prediction = "FAKE"

                confidence = (
                    average_fake_probability
                )

            else:

                prediction = "REAL"

                confidence = (
                    average_real_probability
                )

            # ------------------------------------------
            # FINAL RESULT
            # ------------------------------------------

            result = {

                'prediction':
                    prediction,

                'confidence':
                    float(confidence),

                'fake_probability':
                    float(
                        average_fake_probability
                    ),

                'real_probability':
                    float(
                        average_real_probability
                    ),

                'frames_analyzed':
                    successful_frames,

                'total_frames':
                    total_frames,

                'fps':
                    float(fps),

                'duration':
                    float(duration),

                'selected_frames':
                    number_of_frames
            }

            logger.info(
                f"Final prediction: {prediction}"
            )

            logger.info(
                f"Confidence: {confidence:.2%}"
            )

            return result, None

        except Exception as e:

            logger.error(
                f"Video prediction failed: {e}"
            )

            return None, str(e)


# ======================================================
# INITIALIZE DETECTOR
# ======================================================

detector = DeepfakeDetector()


# ======================================================
# HOME PAGE
# ======================================================

@app.route('/')
def index():

    return render_template(
        'index.html'
    )


# ======================================================
# VIDEO UPLOAD
# ======================================================

@app.route(
    '/upload',
    methods=['POST']
)
def upload_file():

    filepath = None

    try:

        # ----------------------------------------------
        # CHECK FILE
        # ----------------------------------------------

        if 'file' not in request.files:

            return jsonify({
                'error':
                    'No video uploaded'
            }), 400

        file = request.files['file']

        if file.filename == '':

            return jsonify({
                'error':
                    'No video selected'
            }), 400

        # ----------------------------------------------
        # CHECK FILE TYPE
        # ----------------------------------------------

        if not allowed_file(
            file.filename
        ):

            return jsonify({

                'error':
                    'Invalid video type. '
                    'Upload MP4, AVI, MOV, '
                    'or MKV.'
            }), 400

        # ----------------------------------------------
        # SECURE FILE NAME
        # ----------------------------------------------

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        # ----------------------------------------------
        # SAVE VIDEO
        # ----------------------------------------------

        file.save(
            filepath
        )

        logger.info(
            f"Video uploaded: {filename}"
        )

        # ----------------------------------------------
        # PREDICT VIDEO
        # ----------------------------------------------

        result, error = detector.predict_video(
            filepath
        )

        if error:

            return jsonify({
                'error': error
            }), 500

        # ----------------------------------------------
        # RESPONSE
        # ----------------------------------------------

        response = {

            'success':
                True,

            'type':
                'video',

            'prediction':
                result['prediction'],

            'confidence':
                f"{result['confidence']:.1%}",

            'probabilities': {

                'fake':
                    f"{result['fake_probability']:.1%}",

                'real':
                    f"{result['real_probability']:.1%}"
            },

            'frames_analyzed':
                result['frames_analyzed'],

            'total_frames':
                result['total_frames'],

            'fps':
                f"{result['fps']:.2f}",

            'duration':
                f"{result['duration']:.2f}",

            'filename':
                filename
        }

        return jsonify(
            response
        )

    except Exception as e:

        logger.error(
            f"Upload handling failed: {e}"
        )

        return jsonify({

            'error':
                'Internal server error'

        }), 500

    finally:

        # ----------------------------------------------
        # DELETE TEMPORARY VIDEO
        # ----------------------------------------------

        if filepath:

            try:

                if os.path.exists(
                    filepath
                ):

                    os.remove(
                        filepath
                    )

                    logger.info(
                        f"Temporary video deleted: "
                        f"{filepath}"
                    )

            except Exception as e:

                logger.warning(
                    f"Could not delete video: "
                    f"{e}"
                )


# ======================================================
# ALLOWED VIDEO FILE TYPES
# ======================================================

def allowed_file(filename):

    allowed_extensions = {
        'mp4',
        'avi',
        'mov',
        'mkv'
    }

    return (
        '.' in filename
        and
        filename.rsplit(
            '.',
            1
        )[1].lower()
        in allowed_extensions
    )


# ======================================================
# HEALTH CHECK
# ======================================================

@app.route('/health')
def health_check():

    return jsonify({

        'status':
            'healthy',

        'model_loaded':
            detector.model_loaded,

        'version':
            '2.0.0',

        'mode':
            'video-only',

        'supported_formats': [
            'MP4',
            'AVI',
            'MOV',
            'MKV'
        ]
    })


# ======================================================
# START APPLICATION
# ======================================================

if __name__ == '__main__':

    print()
    print(
        "=============================================="
    )
    print(
        "   DEEPFAKE VIDEO DETECTOR"
    )
    print(
        "=============================================="
    )

    print()

    print(
        "Starting Deepfake Video Detector..."
    )

    print(
        "Loading deepfake detection model..."
    )

    # ----------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------

    if detector.load_model():

        print()

        print(
            "Model loaded successfully!"
        )

        print(
            "Video detection mode enabled"
        )

        print(
            "Supported formats: "
            "MP4, AVI, MOV, MKV"
        )

        print()

        print(
            "Starting web server..."
        )

        print(
            "Open your browser and go to:"
        )

        print(
            "http://localhost:5000"
        )

        print()

        print(
            "Upload a video to detect REAL or FAKE"
        )

        print()

        app.run(
            debug=False,
            host='0.0.0.0',
            port=5000
        )

    else:

        print()

        print(
            "Failed to load model!"
        )

        print()

        print(
            "Make sure one of these model "
            "folders exists:"
        )

        print()

        print(
            "- deepfake_detector_savedmodel/"
        )

        print(
            "- hybrid_deepfake_detector_savedmodel/"
        )

        print()

        print(
            "Application was not started."
        )