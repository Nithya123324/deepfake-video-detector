from flask import Flask, render_template, request, jsonify
import os
import io
import base64
from PIL import Image
import numpy as np
import cv2
import requests
import tempfile

app = Flask(__name__)

# Configuration for Vercel deployment
MODEL_URL = "https://your-model-storage.com/deepfake_model"  # Replace with your model URL
MODEL_LOADED = False
MODEL = None

def load_model():
    """Load the model from external storage"""
    global MODEL, MODEL_LOADED
    
    if MODEL_LOADED:
        return MODEL
    
    try:
        # For Vercel, we'll use a placeholder model
        # In production, replace this with actual model loading from external storage
        print("Loading model for Vercel deployment...")
        
        # Placeholder model loading - replace with actual implementation
        # This is a simplified version for demonstration
        MODEL = "placeholder_model"  # Replace with actual model loading
        MODEL_LOADED = True
        
        print("Model loaded successfully!")
        return MODEL
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_deepfake(image_array):
    """Make prediction using the model"""
    try:
        # For Vercel deployment, this is a placeholder
        # Replace with actual model prediction
        import random
        
        # Simulate prediction (replace with actual model inference)
        fake_prob = random.uniform(0.1, 0.9)
        real_prob = 1 - fake_prob
        
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = max(fake_prob, real_prob) * 100
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'real_prob': real_prob,
            'fake_prob': fake_prob,
            'raw_score': fake_prob
        }
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def create_heatmap(image_array):
    """Create a content-aware heatmap"""
    try:
        # Convert to OpenCV format
        img = (image_array[0] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Create content-aware heatmap using edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create heatmap from edges
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        
        # Resize back to original
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Convert back to PIL
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap_rgb)
        
        # Convert to base64
        buffer = io.BytesIO()
        heatmap_pil.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model if not already loaded
        model = load_model()
        if not model:
            return jsonify({'error': 'Model not available'}), 500
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load and preprocess image
        image = Image.open(file.stream)
        image_array = preprocess_image(image)
        
        if image_array is None:
            return jsonify({'error': 'Error preprocessing image'}), 400
        
        # Make prediction
        result = predict_deepfake(image_array)
        
        if result is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        # Create heatmap
        heatmap = create_heatmap(image_array)
        
        return jsonify({
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 2),
            'real_prob': round(result['real_prob'], 4),
            'fake_prob': round(result['fake_prob'], 4),
            'raw_score': round(result['raw_score'], 4),
            'heatmap': heatmap
        })
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
