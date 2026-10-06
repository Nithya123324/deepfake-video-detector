#!/usr/bin/env python3
"""
Fixed debug script to identify the prediction issue
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

def load_and_test_model():
    """Load model and test predictions"""
    print("DEBUG: Loading model...")
    
    # Try loading SavedModel
    model_paths = [
        'deepfake_detector_savedmodel',
        'hybrid_deepfake_detector_savedmodel'
    ]
    
    model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Found model: {model_path}")
            try:
                loaded_model = tf.saved_model.load(model_path)
                print("SavedModel loaded successfully!")
                
                if hasattr(loaded_model, 'signatures'):
                    if 'serving_default' in loaded_model.signatures:
                        inference_func = loaded_model.signatures['serving_default']
                        
                        # Create wrapper
                        def predict_wrapper(x):
                            if not isinstance(x, tf.Tensor):
                                x = tf.convert_to_tensor(x, dtype=tf.float32)
                            
                            result = inference_func(input_1=x)
                            return result['dense_3'].numpy()  # Adjust output key if needed
                        
                        model = predict_wrapper
                        break
                        
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
    
    if model is None:
        print("ERROR: No model could be loaded!")
        return
    
    # Test with different inputs
    print("\nTesting predictions...")
    
    # Test multiple random inputs
    predictions = []
    for i in range(10):
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model(test_input)
        pred_value = float(pred.flatten()[0])
        predictions.append(pred_value)
        classification = 'FAKE' if pred_value > 0.5 else 'REAL'
        print(f"Test {i+1}: {pred_value:.6f} -> {classification}")
    
    # Analyze predictions
    print(f"\nAnalysis:")
    print(f"Min prediction: {min(predictions):.6f}")
    print(f"Max prediction: {max(predictions):.6f}")
    print(f"Mean prediction: {np.mean(predictions):.6f}")
    print(f"Std deviation: {np.std(predictions):.6f}")
    
    # Count classifications
    real_count = sum(1 for p in predictions if p < 0.5)
    fake_count = sum(1 for p in predictions if p >= 0.5)
    print(f"REAL predictions: {real_count}/10")
    print(f"FAKE predictions: {fake_count}/10")
    
    if real_count == 10:
        print("\nPROBLEM IDENTIFIED: Model always predicts REAL!")
        print("Possible causes:")
        print("1. Model was overtrained and converged to always predict majority class")
        print("2. Class imbalance during training")
        print("3. Wrong model weights loaded")
        print("4. Model needs different threshold (try 0.3 instead of 0.5)")
        
        # Test with different threshold
        print(f"\nWith threshold 0.3:")
        real_count_03 = sum(1 for p in predictions if p < 0.3)
        fake_count_03 = sum(1 for p in predictions if p >= 0.3)
        print(f"REAL predictions: {real_count_03}/10")
        print(f"FAKE predictions: {fake_count_03}/10")
        
        print(f"\nWith threshold 0.2:")
        real_count_02 = sum(1 for p in predictions if p < 0.2)
        fake_count_02 = sum(1 for p in predictions if p >= 0.2)
        print(f"REAL predictions: {real_count_02}/10")
        print(f"FAKE predictions: {fake_count_02}/10")

def test_with_actual_image():
    """Test with an actual image if available"""
    print("\nTesting with actual images...")
    
    # Look for any image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for file in os.listdir('.'):
            if file.lower().endswith(ext):
                image_files.append(file)
    
    if not image_files:
        print("No image files found for testing")
        return
    
    # Load model
    try:
        loaded_model = tf.saved_model.load('deepfake_detector_savedmodel')
        inference_func = loaded_model.signatures['serving_default']
        
        def predict_wrapper(x):
            if not isinstance(x, tf.Tensor):
                x = tf.convert_to_tensor(x, dtype=tf.float32)
            result = inference_func(input_1=x)
            return result['dense_3'].numpy()
        
        # Test first image
        image_file = image_files[0]
        print(f"Testing with: {image_file}")
        
        # Load and preprocess image
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Predict
        pred = predict_wrapper(image_batch)
        pred_value = float(pred.flatten()[0])
        classification = 'FAKE' if pred_value > 0.5 else 'REAL'
        
        print(f"Prediction: {pred_value:.6f} -> {classification}")
        
    except Exception as e:
        print(f"Error testing with image: {e}")

if __name__ == '__main__':
    load_and_test_model()
    test_with_actual_image()
