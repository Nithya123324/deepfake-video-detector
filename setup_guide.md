# GTX 1650 Local Training Setup Guide

## Prerequisites for Local Training with 4000 Images

### 1. System Requirements
- **GPU**: GTX 1650 (4GB VRAM) ✅
- **RAM**: Minimum 16GB recommended for 4000 images
- **Storage**: At least 20GB free space
- **OS**: Windows 10/11

### 2. Install CUDA and cuDNN

1. **Install CUDA 11.8** (compatible with TensorFlow 2.13+):
   - Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Choose Windows x86_64, exe (local)
   - Install with default settings

2. **Install cuDNN 8.6** for CUDA 11.x:
   - Download from: https://developer.nvidia.com/cudnn
   - Extract to your CUDA installation directory (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`)

3. **Verify CUDA Installation**:
   ```cmd
   nvcc --version
   nvidia-smi
   ```

### 3. Install Python and Dependencies

1. **Install Python 3.9-3.11** (recommended: 3.10):
   - Download from python.org
   - Make sure to add Python to PATH

2. **Create Virtual Environment**:
   ```cmd
   python -m venv deepfake_env
   deepfake_env\Scripts\activate
   ```

3. **Install TensorFlow GPU**:
   ```cmd
   pip install tensorflow-gpu==2.13.0
   ```

4. **Verify GPU Detection**:
   ```python
   import tensorflow as tf
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))
   ```

### 4. Dataset Organization

Organize your 4000 images in this structure:
```
your_dataset/
├── real/          (2000 real images)
└── fake/          (2000 fake images)
```

### 5. Memory Optimization for GTX 1650

**Key Settings for 4GB VRAM**:
- Batch size: 8-16 (start with 8)
- Image size: 224x224
- Mixed precision training: Enabled
- Memory growth: Enabled
- Gradient accumulation: If needed

### 6. Training Time Estimation

With GTX 1650 and 4000 images:
- **Per epoch**: ~15-25 minutes
- **Total training (20 epochs)**: ~5-8 hours
- **Recommended**: Train overnight or during free time

### 7. Monitoring Tools

Install monitoring tools:
```cmd
pip install nvidia-ml-py3 psutil
```

### 8. Troubleshooting Common Issues

**Out of Memory (OOM) Errors**:
- Reduce batch size to 4-8
- Enable memory growth
- Use gradient accumulation

**Slow Training**:
- Verify GPU usage with `nvidia-smi`
- Check CPU bottlenecks
- Ensure SSD storage for faster data loading

**CUDA Errors**:
- Reinstall CUDA/cuDNN
- Check TensorFlow-GPU compatibility
- Restart system after installation
