#!/usr/bin/env python3
"""
Simple test of the fixed web application
"""

import numpy as np
from PIL import Image
from app import DeepfakeDetector

def test_predictions():
    """Test the fixed predictions"""
    print("Testing fixed deepfake detector...")
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Load model
    if detector.load_model():
        print("Model loaded successfully!")
    else:
        print("Model loading failed!")
        return
    
    # Test with different images
    print("\nTesting predictions:")
    
    predictions = []
    classifications = []
    
    for i in range(5):
        # Create random test image
        test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_image)
        
        result, error = detector.predict(test_pil)
        if not error:
            predictions.append(result['raw_score'])
            classifications.append(result['prediction'])
            print(f"Test {i+1}: {result['raw_score']:.6f} -> {result['prediction']} ({result['confidence']:.1%})")
        else:
            print(f"Test {i+1}: ERROR - {error}")
    
    if predictions:
        real_count = classifications.count('REAL')
        fake_count = classifications.count('FAKE')
        
        print(f"\nResults:")
        print(f"REAL predictions: {real_count}/5")
        print(f"FAKE predictions: {fake_count}/5")
        print(f"Range: {min(predictions):.6f} to {max(predictions):.6f}")
        
        if real_count == 5:
            print("ISSUE: Still predicting all REAL")
        elif fake_count == 5:
            print("ISSUE: Predicting all FAKE")
        else:
            print("SUCCESS: Good variety in predictions!")

if __name__ == '__main__':
    test_predictions()
