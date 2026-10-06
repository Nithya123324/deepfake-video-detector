#!/usr/bin/env python3
"""
Simple debug script to test the deepfake model predictions
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
                        
                        # Get input/output info
                        input_spec = inference_func.structured_input_signature[1]
                        print(f"Input spec: {input_spec}")
                        
                        # Create wrapper
                        def predict_wrapper(x):
                            if not isinstance(x, tf.Tensor):
                                x = tf.convert_to_tensor(x, dtype=tf.float32)
                            
                            input_keys = list(input_spec.keys())
                            input_key = input_keys[0] if input_keys else 'input_1'
                            
                            result = inference_func(**{input_key: x})
                            output_keys = list(result.keys())
                            output_key = output_keys[0] if output_keys else 'output_1'
                            
                            return result[output_key].numpy()
                        
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
    
    # Test 1: All zeros (should be fake)
    test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    pred = model(test_input)
    print(f"All zeros input: {pred} -> {'FAKE' if pred[0] > 0.5 else 'REAL'}")
    
    # Test 2: All ones (should be real)
    test_input = np.ones((1, 224, 224, 3), dtype=np.float32)
    pred = model(test_input)
    print(f"All ones input: {pred} -> {'FAKE' if pred[0] > 0.5 else 'REAL'}")
    
    # Test 3: Random noise
    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    pred = model(test_input)
    print(f"Random input: {pred} -> {'FAKE' if pred[0] > 0.5 else 'REAL'}")
    
    # Test 4: Check if model always returns same value
    predictions = []
    for i in range(5):
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model(test_input)
        predictions.append(float(pred[0]))
        print(f"Test {i+1}: {pred[0]:.6f}")
    
    # Check variance
    pred_variance = np.var(predictions)
    print(f"\nPrediction variance: {pred_variance:.8f}")
    
    if pred_variance < 1e-6:
        print("WARNING: Model always returns the same prediction!")
        print("This suggests the model is stuck or has converged to always predict one class.")
        
        # Check the actual prediction value
        avg_pred = np.mean(predictions)
        print(f"Average prediction: {avg_pred:.6f}")
        
        if avg_pred < 0.1:
            print("Model always predicts REAL (values close to 0)")
        elif avg_pred > 0.9:
            print("Model always predicts FAKE (values close to 1)")
        else:
            print(f"Model predictions centered around {avg_pred:.3f}")
    else:
        print("Model shows variation in predictions - this is good!")

if __name__ == '__main__':
    load_and_test_model()
