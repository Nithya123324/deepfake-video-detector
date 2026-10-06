#!/usr/bin/env python3
"""
Deepfake Detector Web Application Launcher
Easy startup script for the web interface
"""

import os
import sys
import subprocess
import platform

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'tensorflow', 'numpy', 'pillow', 'opencv-python', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'web_requirements.txt'])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    model_paths = [
        'deepfake_detector_savedmodel',
        'hybrid_deepfake_detector_savedmodel',
        'deepfake_detector_weights_80_94.h5'
    ]
    
    found_models = []
    for path in model_paths:
        if os.path.exists(path):
            found_models.append(path)
    
    return found_models

def main():
    """Main launcher function"""
    print("🚀 Deepfake Detector Web Application Launcher")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check requirements
    print("\n📋 Checking requirements...")
    missing = check_requirements()
    
    if missing:
        print(f"⚠️ Missing packages: {', '.join(missing)}")
        print("🔧 Installing missing packages...")
        
        if not install_requirements():
            print("❌ Failed to install requirements!")
            print("💡 Try running manually: pip install -r web_requirements.txt")
            sys.exit(1)
    else:
        print("✅ All required packages are installed!")
    
    # Check model files
    print("\n🤖 Checking model files...")
    models = check_model_files()
    
    if not models:
        print("❌ No model files found!")
        print("💡 Make sure you have one of these files/folders:")
        print("   - deepfake_detector_savedmodel/")
        print("   - hybrid_deepfake_detector_savedmodel/")
        print("   - deepfake_detector_weights_80_94.h5")
        print("\n🔧 Copy your trained model files to this directory and try again.")
        sys.exit(1)
    else:
        print(f"✅ Found model files: {', '.join(models)}")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("\n🌐 Starting web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎯 Upload an image to detect if it's real or fake!")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask application
    try:
        from app import app, detector
        
        # Load model
        if not detector.model_loaded:
            print("📂 Loading model...")
            if detector.load_model():
                print("✅ Model loaded successfully!")
            else:
                print("❌ Failed to load model!")
                sys.exit(1)
        
        # Run the app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down web application...")
        print("✅ Server stopped successfully!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("💡 Make sure all files are in place and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()
