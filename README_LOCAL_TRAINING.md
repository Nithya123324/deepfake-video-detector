# GTX 1650 Local Training Guide - 4000 Images DeepFake Detection

## 🚀 Quick Setup

### 1. Prerequisites
- **GPU**: GTX 1650 (4GB VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB free space
- **OS**: Windows 10/11

### 2. Installation Steps

```bash
# 1. Create virtual environment
python -m venv deepfake_env
deepfake_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test setup
python quick_start.py

# 4. Run training
python Real_Dataset_Hybrid_DeepFake_Detector.py
```

## 📁 Dataset Organization

Structure your 4000 images like this:
```
your_dataset/
├── real/          (2000 real images)
└── fake/          (2000 fake images)
```

**Supported formats**: .jpg, .jpeg, .png, .bmp, .tiff

## 🎯 GTX 1650 Optimizations

### Memory Optimization
- ✅ **Batch size**: 8 (optimized for 4GB VRAM)
- ✅ **Mixed precision**: FP16 training (2x speedup)
- ✅ **Memory growth**: Prevents allocation errors
- ✅ **Memory limit**: 3.5GB (reserves 0.5GB for system)

### Performance Features
- ✅ **Data prefetching**: Faster data loading
- ✅ **Optimized augmentation**: Balanced speed vs accuracy
- ✅ **GPU monitoring**: Real-time memory tracking
- ✅ **Automatic cleanup**: Memory management between epochs

### Training Parameters
- **Epochs**: 25 (increased for larger dataset)
- **Learning rate**: 0.0002 (optimized start)
- **Early stopping**: 8 epochs patience
- **Image size**: 224x224
- **Architecture**: Hybrid CNN (EfficientNet) + ViT

## ⏱️ Training Time Estimates

| Dataset Size | Time per Epoch | Total Time (25 epochs) |
|-------------|----------------|------------------------|
| 1000 images | ~6 minutes     | ~2.5 hours            |
| 2000 images | ~12 minutes    | ~5 hours              |
| 4000 images | ~20 minutes    | ~8 hours              |

## 🔧 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in the code:
BATCH_SIZE = 4  # Change from 8 to 4
```

### Slow Training
1. Check GPU usage: `nvidia-smi`
2. Verify SSD storage (not HDD)
3. Close other GPU applications

### CUDA Errors
1. Reinstall CUDA 11.8
2. Reinstall cuDNN 8.6
3. Restart system

## 📊 Expected Results

With 4000 images, you should achieve:
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Test Accuracy**: 85-90%

## 🎛️ Advanced Configuration

### Increase Batch Size (if you have headroom)
```python
# In the code, try:
BATCH_SIZE = 12  # Test if GPU can handle it
```

### Longer Training
```python
# Increase epochs for better accuracy:
self.epochs = 35  # In GTX1650TrainingConfig class
```

### Fine-tuning
```python
# Unfreeze more layers for fine-tuning:
for layer in cnn_base.layers[-20:]:  # Instead of -10
    layer.trainable = True
```

## 📁 Output Files

After training, you'll get:
- `best_deepfake_detector_gtx1650.h5` - Complete trained model
- `training_history_real_data.png` - Training curves
- `confusion_matrix_real_data.png` - Performance matrix
- `roc_curve_real_data.png` - ROC curve
- `gradcam_explanations_real_data.png` - AI explanations

## 🚀 Model Architecture (Preserved)

✅ **Hybrid Model**: CNN (EfficientNet) + Vision Transformer
✅ **Explainable AI**: Grad-CAM + SHAP integration
✅ **Transfer Learning**: Pre-trained on ImageNet
✅ **Custom Head**: Optimized for deepfake detection

## 💡 Tips for Better Results

1. **Data Quality**: Ensure high-quality, diverse images
2. **Class Balance**: Keep real/fake ratio close to 50/50
3. **Training Time**: Let it train overnight for best results
4. **Validation**: Monitor validation loss to prevent overfitting
5. **Testing**: Always test on unseen data

## 🆘 Support

If you encounter issues:
1. Run `python quick_start.py` to diagnose problems
2. Check GPU memory with `nvidia-smi`
3. Verify dataset structure
4. Check CUDA/cuDNN installation

---

**Note**: This configuration maintains your original hybrid CNN+ViT architecture with explainable AI features while optimizing for GTX 1650 performance with 4000 images.
