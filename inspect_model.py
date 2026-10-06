#!/usr/bin/env python3
"""
Inspect the model structure to understand the output keys
"""

import tensorflow as tf
import numpy as np

def inspect_model():
    """Inspect the SavedModel structure"""
    print("Inspecting model structure...")
    
    try:
        # Load the model
        loaded_model = tf.saved_model.load('deepfake_detector_savedmodel')
        print("Model loaded successfully!")
        
        # Check signatures
        if hasattr(loaded_model, 'signatures'):
            print(f"Available signatures: {list(loaded_model.signatures.keys())}")
            
            if 'serving_default' in loaded_model.signatures:
                inference_func = loaded_model.signatures['serving_default']
                
                # Get input and output specs
                input_spec = inference_func.structured_input_signature[1]
                output_spec = inference_func.structured_outputs
                
                print(f"Input spec: {input_spec}")
                print(f"Output spec: {output_spec}")
                
                # Test with dummy input to see actual output
                dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                dummy_tensor = tf.convert_to_tensor(dummy_input)
                
                # Get input key
                input_keys = list(input_spec.keys())
                input_key = input_keys[0] if input_keys else 'input_1'
                print(f"Using input key: {input_key}")
                
                # Make prediction
                result = inference_func(**{input_key: dummy_tensor})
                
                print(f"Result type: {type(result)}")
                print(f"Result keys: {list(result.keys())}")
                
                for key, value in result.items():
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"    Sample values: {value.numpy().flatten()[:5]}")
                
                # Find the main output (usually the one with shape (batch_size, 1))
                main_output_key = None
                for key, value in result.items():
                    if len(value.shape) == 2 and value.shape[1] == 1:
                        main_output_key = key
                        break
                
                if main_output_key:
                    print(f"Main output key appears to be: {main_output_key}")
                else:
                    print("Could not identify main output key")
                    # Use the first key as fallback
                    main_output_key = list(result.keys())[0]
                    print(f"Using first key as fallback: {main_output_key}")
                
                return input_key, main_output_key
                
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None, None

def test_corrected_prediction():
    """Test prediction with correct keys"""
    input_key, output_key = inspect_model()
    
    if input_key is None or output_key is None:
        print("Could not determine correct keys")
        return
    
    print(f"\nTesting with keys: input='{input_key}', output='{output_key}'")
    
    try:
        # Load model
        loaded_model = tf.saved_model.load('deepfake_detector_savedmodel')
        inference_func = loaded_model.signatures['serving_default']
        
        # Test multiple predictions
        predictions = []
        for i in range(5):
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_tensor = tf.convert_to_tensor(test_input)
            
            result = inference_func(**{input_key: test_tensor})
            pred_value = float(result[output_key].numpy().flatten()[0])
            predictions.append(pred_value)
            
            classification = 'FAKE' if pred_value > 0.5 else 'REAL'
            print(f"Test {i+1}: {pred_value:.6f} -> {classification}")
        
        # Analysis
        print(f"\nPrediction range: {min(predictions):.6f} to {max(predictions):.6f}")
        print(f"Mean: {np.mean(predictions):.6f}")
        
        real_count = sum(1 for p in predictions if p < 0.5)
        fake_count = len(predictions) - real_count
        
        print(f"REAL: {real_count}/{len(predictions)}")
        print(f"FAKE: {fake_count}/{len(predictions)}")
        
        if real_count == len(predictions):
            print("\nISSUE CONFIRMED: Model always predicts REAL")
            print("The model predictions are all below 0.5")
            
            # Suggest solutions
            print("\nPOSSIBLE SOLUTIONS:")
            print("1. Adjust threshold to 0.3 or lower")
            print("2. Use a different model checkpoint")
            print("3. Check if model was trained with different preprocessing")
            
            # Test different thresholds
            for threshold in [0.4, 0.3, 0.2]:
                real_t = sum(1 for p in predictions if p < threshold)
                fake_t = len(predictions) - real_t
                print(f"With threshold {threshold}: REAL={real_t}, FAKE={fake_t}")
        
    except Exception as e:
        print(f"Error in corrected prediction test: {e}")

if __name__ == '__main__':
    test_corrected_prediction()
