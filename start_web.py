#!/usr/bin/env python3
"""
Simple startup script for the deepfake detector web interface
No double loading issues
"""

import os
from app import app, detector

def main():
    """Start the web application"""
    print("🚀 Starting Deepfake Detector Web Interface...")
    
    # Check if model files exist
    model_files = [
        'deepfake_detector_savedmodel',
        'hybrid_deepfake_detector_savedmodel'
    ]
    
    model_found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            model_found = True
            break
    
    if not model_found:
        print("❌ No model files found!")
        print("💡 Make sure you have your trained model files in this directory")
        return
    
    # Load model once
    print("📂 Loading model...")
    if detector.load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model!")
        return
    
    # Create upload directory
    os.makedirs('uploads', exist_ok=True)
    
    # Start server
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎯 Upload an image to detect if it's real or fake!")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")

if __name__ == '__main__':
    main()

