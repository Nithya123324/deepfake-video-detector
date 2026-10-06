#!/usr/bin/env python3
"""
Test the fixed web application model loading
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Import the fixed detector class
from app import DeepfakeDetector

def test_fixed_detector():
    """Test the fixed detector"""
    print("Testing fixed deepfake detector...")
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Load model
    print("Loading model...")
    if detector.load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model loading failed!")
        return
    
    # Create test images
    print("\nTesting predictions...")
    
    # Test 1: Create a simple test image
    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_image)
    
    result, error = detector.predict(test_pil)
    if error:
        print(f"❌ Prediction failed: {error}")
    else:
        print(f"✅ Test 1 - Random image:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Raw score: {result['raw_score']:.6f}")
    
    # Test 2: White image
    white_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    white_pil = Image.fromarray(white_image)
    
    result, error = detector.predict(white_pil)
    if error:
        print(f"❌ Prediction failed: {error}")
    else:
        print(f"✅ Test 2 - White image:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Raw score: {result['raw_score']:.6f}")
    
    # Test 3: Black image
    black_image = np.zeros((224, 224, 3), dtype=np.uint8)
    black_pil = Image.fromarray(black_image)
    
    result, error = detector.predict(black_pil)
    if error:
        print(f"❌ Prediction failed: {error}")
    else:
        print(f"✅ Test 3 - Black image:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Raw score: {result['raw_score']:.6f}")
    
    # Test multiple random images to check variety
    print(f"\nTesting variety with 10 random images:")
    predictions = []
    classifications = []
    
    for i in range(10):
        test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_image)
        
        result, error = detector.predict(test_pil)
        if not error:
            predictions.append(result['raw_score'])
            classifications.append(result['prediction'])
            print(f"   Test {i+1}: {result['raw_score']:.6f} -> {result['prediction']}")
    
    if predictions:
        real_count = classifications.count('REAL')
        fake_count = classifications.count('FAKE')
        
        print(f"\nSummary:")
        print(f"   REAL predictions: {real_count}/10")
        print(f"   FAKE predictions: {fake_count}/10")
        print(f"   Prediction range: {min(predictions):.6f} to {max(predictions):.6f}")
        print(f"   Mean prediction: {np.mean(predictions):.6f}")
        
        if real_count == 10:
            print("⚠️ Still predicting all REAL - may need threshold adjustment")
        elif fake_count == 10:
            print("⚠️ Predicting all FAKE - may need threshold adjustment")
        else:
            print("✅ Good variety in predictions!")

if __name__ == '__main__':
    test_fixed_detector()
