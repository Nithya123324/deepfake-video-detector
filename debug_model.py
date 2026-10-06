#!/usr/bin/env python3
"""
Debug script to test the deepfake model and identify prediction issues
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

def load_model_debug():
    """Load model with detailed debugging"""
    print("🔍 Debugging model loading...")
    
    # Try loading SavedModel
    model_paths = [
        'deepfake_detector_savedmodel',
        'hybrid_deepfake_detector_savedmodel'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"📂 Found model: {model_path}")
            try:
                loaded_model = tf.saved_model.load(model_path)
                print("✅ SavedModel loaded successfully!")
                
                # Check signatures
                if hasattr(loaded_model, 'signatures'):
                    print(f"📋 Available signatures: {list(loaded_model.signatures.keys())}")
                    
                    if 'serving_default' in loaded_model.signatures:
                        inference_func = loaded_model.signatures['serving_default']
                        print("✅ Found serving_default signature")
                        
                        # Check input/output specs
                        input_spec = inference_func.structured_input_signature[1]
                        output_spec = inference_func.structured_outputs
                        
                        print(f"📥 Input spec: {input_spec}")
                        print(f"📤 Output spec: {output_spec}")
                        
                        # Create wrapper function
                        def predict_wrapper(x):
                            if not isinstance(x, tf.Tensor):
                                x = tf.convert_to_tensor(x, dtype=tf.float32)
                            
                            input_keys = list(input_spec.keys())
                            input_key = input_keys[0] if input_keys else 'input_1'
                            
                            print(f"🔑 Using input key: {input_key}")
                            print(f"📊 Input shape: {x.shape}")
                            
                            result = inference_func(**{input_key: x})
                            
                            output_keys = list(result.keys())
                            output_key = output_keys[0] if output_keys else 'output_1'
                            
                            print(f"🔑 Using output key: {output_key}")
                            print(f"📊 Raw output: {result[output_key]}")
                            
                            return result[output_key].numpy()
                        
                        return predict_wrapper, True
                        
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                continue
    
    print("❌ No model could be loaded!")
    return None, False

def preprocess_image_debug(image_path):
    """Preprocess image with debugging"""
    print(f"\n🖼️ Processing image: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    print(f"📊 Original image: {image.size}, mode: {image.mode}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
        print("🔄 Converted to RGB")
    
    # Convert to numpy
    image_np = np.array(image)
    print(f"📊 NumPy shape: {image_np.shape}, dtype: {image_np.dtype}")
    print(f"📊 Value range: {image_np.min()} - {image_np.max()}")
    
    # Resize
    image_resized = cv2.resize(image_np, (224, 224))
    print(f"📊 Resized shape: {image_resized.shape}")
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    print(f"📊 Normalized range: {image_normalized.min():.4f} - {image_normalized.max():.4f}")
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    print(f"📊 Final shape: {image_batch.shape}")
    
    return image_batch

def test_predictions(model, test_images):
    """Test predictions on multiple images"""
    print("\n🧪 Testing predictions...")
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue
            
        print(f"\n--- Test {i+1}: {os.path.basename(image_path)} ---")
        
        # Preprocess
        processed_image = preprocess_image_debug(image_path)
        
        # Predict
        try:
            prediction = model(processed_image)
            print(f"🎯 Raw prediction: {prediction}")
            print(f"📊 Prediction shape: {prediction.shape}")
            print(f"📊 Prediction dtype: {prediction.dtype}")
            
            # Extract value
            pred_value = float(prediction.flatten()[0])
            print(f"🔢 Extracted value: {pred_value}")
            
            # Classify
            if pred_value > 0.5:
                predicted_class = "FAKE"
                confidence = pred_value
            else:
                predicted_class = "REAL"
                confidence = 1 - pred_value
            
            print(f"🏷️ Classification: {predicted_class}")
            print(f"📈 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")

def create_test_images():
    """Create simple test images for debugging"""
    print("\n🎨 Creating test images...")
    
    # Create a white image (should be more likely to be classified as real)
    white_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    Image.fromarray(white_image).save('test_white.jpg')
    print("✅ Created test_white.jpg")
    
    # Create a black image (should be more likely to be classified as fake)
    black_image = np.zeros((224, 224, 3), dtype=np.uint8)
    Image.fromarray(black_image).save('test_black.jpg')
    print("✅ Created test_black.jpg")
    
    # Create a random noise image (should be more likely to be classified as fake)
    noise_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(noise_image).save('test_noise.jpg')
    print("✅ Created test_noise.jpg")
    
    return ['test_white.jpg', 'test_black.jpg', 'test_noise.jpg']

def check_model_bias():
    """Check if model has prediction bias"""
    print("\n🔍 Checking for model bias...")
    
    # Load model
    model, loaded = load_model_debug()
    if not loaded:
        print("❌ Cannot check bias - model not loaded")
        return
    
    # Create test images
    test_images = create_test_images()
    
    # Test predictions
    test_predictions(model, test_images)
    
    # Clean up test images
    for img in test_images:
        try:
            os.remove(img)
        except:
            pass

def main():
    """Main debugging function"""
    print("🔍 DEEPFAKE MODEL DEBUG TOOL")
    print("=" * 50)
    
    # Check if model files exist
    print("\n📁 Checking model files...")
    model_files = [
        'deepfake_detector_savedmodel',
        'hybrid_deepfake_detector_savedmodel',
        'deepfake_detector_weights_80_94.h5'
    ]
    
    found_files = []
    for file in model_files:
        if os.path.exists(file):
            found_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    if not found_files:
        print("\n❌ No model files found!")
        return
    
    # Run bias check
    check_model_bias()
    
    print("\n" + "=" * 50)
    print("🎯 DEBUG COMPLETE!")
    print("\nIf all predictions show the same class, possible issues:")
    print("1. Model was trained with class imbalance")
    print("2. Model converged to always predict majority class")
    print("3. Preprocessing doesn't match training preprocessing")
    print("4. Model weights are from wrong epoch")
    print("5. Threshold needs adjustment (try 0.3 or 0.7 instead of 0.5)")

if __name__ == '__main__':
    main()
