# Methodology

## Overview

This document describes the methodology for building a chest X-ray classification system capable of distinguishing between Normal, Pneumonia, and Tuberculosis cases.

## Data Pipeline

### 1. Data Acquisition
- **Source**: Kaggle dataset (muhammadrehan00/chest-xray-dataset)
- **Download**: Automated via `kagglehub` library
- **Storage**: Raw data in `data/raw/`, processed splits in `data/processed/`

### 2. Exploratory Data Analysis
Key analyses performed:
- Class distribution and imbalance assessment
- Image property analysis (size, format, intensity distributions)
- Sample visualization across classes

### 3. Data Cleaning & Validation

Before training, images are validated and cleaned:

**Quality Checks**:
- Verify image integrity (detect corrupted files)
- Filter images with dimensions < 100×100 pixels
- Flag images with extreme aspect ratios (> 3:1)
- Identify and remove near-duplicate images using perceptual hashing

**Duplicate Detection**:
```python
from imagehash import phash
# Images with perceptual hash distance < 5 are flagged as duplicates
```

**Class Imbalance Handling**:
| Technique | Implementation |
|-----------|----------------|
| Weighted Loss | Inverse frequency class weights in CrossEntropyLoss |
| Weighted Sampling | `WeightedRandomSampler` balances class distribution per batch |
| Stratified Splits | Train/val/test splits preserve class proportions |

**Optional Enhancements**:
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for enhanced lung visibility
- **Label Noise Detection**: Visual inspection or pretrained model confidence checks

### 4. Preprocessing Pipeline
```
Raw Image → Load → Convert to RGB → Resize → Augment (train only) → Normalize → Tensor
```

**Training Augmentations**:
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness/contrast)
- Random affine transforms
- Random crop with resize

**Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Medical Imaging Considerations**:
- Grayscale X-rays converted to 3-channel RGB for ImageNet pretrained models
- 224×224 resolution standard for EfficientNet/ResNet families
- Optional CLAHE preprocessing can enhance lung structure visibility:
  ```python
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  ```

## Model Architecture

### Model Selection Rationale

| Model | Parameters | Why Consider | Best For |
|-------|------------|--------------|----------|
| **EfficientNet-B0** ✓ | ~5.3M | Excellent accuracy/efficiency, strong transfer learning | Default choice |
| **EfficientNet-B2** | ~9.2M | Higher capacity for complex patterns | More compute budget |
| **DenseNet-121** | ~8M | Feature reuse, proven on CheXpert/ChestX-ray14 | Interpretability (GradCAM) |
| **ConvNeXt-Tiny** | ~28M | State-of-the-art modern architecture | Maximum accuracy |
| **ResNet-50** | ~25M | Well-studied, reliable baseline | Comparison/reproducibility |

**Primary Choice: EfficientNet-B0**
- Compound scaling balances depth, width, and resolution
- Efficient for inference (important for deployment)
- Strong performance on medical imaging benchmarks
- Easy to swap via `timm` library if needed

### Transfer Learning Approach
We use transfer learning with models pretrained on ImageNet:

1. **Backbone**: EfficientNet-B0 (default)
   - Modern, efficient architecture
   - Good accuracy/speed tradeoff
   - Pretrained on ImageNet

2. **Classification Head**:
   ```
   GlobalAvgPool → BatchNorm → Dropout → Dense(256) → ReLU → BatchNorm → Dropout → Dense(3)
   ```

### Progressive Unfreezing
1. **Epochs 1-5**: Train only classification head (backbone frozen)
2. **Epochs 6+**: Unfreeze backbone for fine-tuning

## Training Strategy

### Optimizer
- **AdamW** with weight decay (0.01)
- Initial learning rate: 0.001

### Learning Rate Schedule
- Warmup: 3 epochs (linear ramp from 0.1x to 1x)
- Cosine annealing after warmup

### Loss Function
- Cross-entropy with label smoothing (0.1)
- Class weights for imbalanced data

### Regularization
- Dropout (0.3 in head)
- Weight decay (L2)
- Early stopping (patience=10)
- Gradient clipping (max_norm=1.0)

### Mixed Precision Training
- FP16 training on CUDA devices
- ~2x speedup with minimal accuracy loss

## Evaluation

### Metrics
- **Primary**: Macro F1 Score
- **Secondary**: Accuracy, Per-class Precision/Recall
- **ROC-AUC**: One-vs-Rest for each class

### Visualization
- Confusion matrix
- ROC curves
- Grad-CAM attention maps

## Deployment

### API Design
- FastAPI for high-performance async API
- Endpoints: `/predict`, `/predict/batch`, `/health`
- Input: Image file (JPEG, PNG)
- Output: JSON with class, confidence, probabilities

### Containerization
- Multi-stage Docker build
- Non-root user for security
- Health checks included

## Reproducibility

### Seed Control
All random sources seeded:
- Python `random`
- NumPy
- PyTorch (CPU and CUDA)
- PYTHONHASHSEED environment variable
- Deterministic cuDNN operations

### Configuration Management
- All hyperparameters in YAML config files
- Configs saved with model checkpoints
- Training history logged to JSON

## Future Improvements

1. **Model Ensemble**: Combine predictions from multiple architectures
2. **Test-Time Augmentation**: Average predictions over augmented inputs
3. **Uncertainty Quantification**: Monte Carlo dropout for confidence estimates
4. **Explainability**: More advanced attention visualization (Attention Rollout, etc.)
5. **Active Learning**: Identify uncertain samples for expert labeling

