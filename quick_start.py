#!/usr/bin/env python3
"""
Quick Start Script for GTX 1650 DeepFake Detection Training
Run this first to test your setup before running the main training script
"""

import os
import sys

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.9+")
        return False

def check_gpu_setup():
    """Test GPU setup and CUDA availability"""
    print("🔍 Checking GPU setup...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} installed")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU(s) detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            # Test GPU memory setup
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("✅ GPU memory growth enabled")
            except Exception as e:
                print(f"⚠️ GPU memory setup warning: {e}")
                
            return True
        else:
            print("❌ No GPU detected!")
            print("💡 Check CUDA installation and GPU drivers")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed!")
        print("💡 Run: pip install tensorflow-gpu==2.13.0")
        return False
    except Exception as e:
        print(f"❌ GPU setup error: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        ('tensorflow', 'tensorflow-gpu==2.13.0'),
        ('transformers', 'transformers==4.33.2'),
        ('opencv-python', 'opencv-python==4.8.1.78'),
        ('scikit-learn', 'scikit-learn==1.3.0'),
        ('matplotlib', 'matplotlib==3.7.2'),
        ('seaborn', 'seaborn==0.12.2'),
        ('shap', 'shap==0.42.1'),
        ('pillow', 'Pillow==10.0.1'),
    ]
    
    missing_packages = []
    
    for package_name, install_command in required_packages:
        try:
            __import__(package_name.replace('-', '_'))
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(install_command)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print("pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All dependencies installed!")
        return True

def check_dataset_structure():
    """Guide user to set up dataset properly"""
    print("\n📁 Dataset setup guide:")
    print("Organize your 4000 images like this:")
    print("""
    your_dataset/
    ├── real/     (2000 real images: .jpg, .png, .jpeg)
    └── fake/     (2000 fake images: .jpg, .png, .jpeg)
    """)
    
    # Try to find dataset
    possible_paths = [
        './dataset',
        './data', 
        '../dataset',
        os.path.expanduser('~/dataset'),
        os.path.expanduser('~/Downloads/dataset')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            real_path = os.path.join(path, 'real')
            fake_path = os.path.join(path, 'fake')
            
            if os.path.exists(real_path) and os.path.exists(fake_path):
                real_count = len([f for f in os.listdir(real_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                fake_count = len([f for f in os.listdir(fake_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                print(f"✅ Found dataset at: {path}")
                print(f"   Real images: {real_count}")
                print(f"   Fake images: {fake_count}")
                print(f"   Total: {real_count + fake_count}")
                
                if real_count + fake_count >= 1000:
                    return True
                else:
                    print("⚠️ Dataset seems small - need at least 1000 images")
    
    print("❌ No properly structured dataset found")
    print("💡 Place your dataset in one of these locations:")
    for path in possible_paths[:3]:
        print(f"   {path}")
    
    return False

def estimate_training_requirements():
    """Estimate training time and requirements"""
    print("\n⏱️ Training Estimates for GTX 1650:")
    print("   Batch size: 8 (optimized for 4GB VRAM)")
    print("   Time per epoch: ~15-25 minutes (for 4000 images)")
    print("   Total training time: ~6-10 hours (25 epochs)")
    print("   GPU memory usage: ~3.5GB")
    print("   Disk space needed: ~5GB (for model files)")

def main():
    """Run all setup checks"""
    print("🚀 GTX 1650 DeepFake Detection Setup Checker")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU & CUDA Setup", check_gpu_setup), 
        ("Dependencies", check_dependencies),
        ("Dataset Structure", check_dataset_structure),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        if not check_func():
            all_passed = False
    
    estimate_training_requirements()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Setup Complete! Ready to train your model.")
        print("💡 Run the main script: python Real_Dataset_Hybrid_DeepFake_Detector.py")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        print("💡 Refer to setup_guide.md for detailed instructions")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
