# Chest X-Ray Classification System

## Executive Summary

This project demonstrates a complete end-to-end deep learning pipeline for medical image classification, specifically classifying chest X-rays into three diagnostic categories: **Normal**, **Pneumonia**, and **Tuberculosis**.

The system achieves **90.8% macro AUC** and **77.1% accuracy** on a held-out test set of 2,569 images, with particularly strong performance on Pneumonia detection (98.7% AUC, 100% recall).

---

## Key Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 77.1% |
| **Macro AUC-ROC** | 90.8% |
| **Test Samples** | 2,569 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | AUC | Support |
|-------|-----------|--------|----------|-----|---------|
| Normal | 64.8% | 80.4% | 71.8% | 83.9% | 925 |
| Pneumonia | 77.5% | 100% | 87.3% | 98.7% | 580 |
| Tuberculosis | 97.5% | 61.7% | 75.5% | 89.7% | 1,064 |

**Key Insight**: The model shows excellent Pneumonia detection (zero missed cases) and high Tuberculosis precision (very few false positives), making it particularly useful as a screening tool where catching potential Pneumonia cases is critical.

---

## Technical Approach

### Architecture
- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Classification Head**: BatchNorm → Dropout(0.3) → Dense(256) → ReLU → BatchNorm → Dropout(0.15) → Dense(3)
- **Total Parameters**: ~5.3M

### Training Strategy
- **Transfer Learning**: Progressive unfreezing (classifier first, then backbone)
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: Cosine annealing with warmup (3 epochs)
- **Loss**: Cross-entropy with label smoothing (0.1) and class weights
- **Regularization**: Dropout, weight decay, early stopping, gradient clipping
- **Mixed Precision**: FP16 training for efficiency

### Data Pipeline
- **Source**: Kaggle chest-xray-dataset
- **Preprocessing**: Resize → Center crop → Normalize (ImageNet stats)
- **Augmentation**: Random flip, rotation (±15°), color jitter, affine transforms

---

## Explainability

The system includes **Grad-CAM visualizations** showing which regions of the X-ray influence the model's decision. This is critical for medical applications where interpretability builds trust with clinicians.

![Grad-CAM Analysis](../reports/figures/gradcam_analysis.png)

The attention maps show the model correctly focuses on:
- Lung fields for Normal cases
- Infiltrate regions for Pneumonia
- Apical/upper lobe regions typical of Tuberculosis

---

## Deployment

### REST API
A production-ready FastAPI service with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Single image classification |
| `/predict/batch` | POST | Batch classification (up to 10 images) |
| `/predict/explain` | POST | Classification with Grad-CAM visualization |

### Docker Support
```bash
docker build -t xray-classifier .
docker run -p 8000:8000 xray-classifier
```

### API Response Example
```json
{
  "predicted_class": "Pneumonia",
  "confidence": 0.94,
  "probabilities": {
    "Normal": 0.03,
    "Pneumonia": 0.94,
    "Tuberculosis": 0.03
  },
  "inference_time_ms": 45.2
}
```

---

## Project Structure

```
├── configs/           # YAML configuration files
├── data/              # Raw and processed datasets
├── docs/              # Detailed methodology documentation
├── models/            # Trained model checkpoints
├── notebooks/         # Exploratory data analysis
├── reports/figures/   # Evaluation visualizations
├── src/
│   ├── api/           # FastAPI deployment
│   ├── data/          # Data loading & transforms
│   ├── models/        # Architecture & training
│   └── visualization/ # Grad-CAM & plotting
└── tests/             # Unit tests
```

---

## Reproducibility

- **Deterministic Training**: All random seeds fixed (Python, NumPy, PyTorch, CUDA)
- **Configuration Management**: All hyperparameters in version-controlled YAML
- **Checkpointing**: Full model state + config saved together

---

## How to Run

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data
python -m src.data.download_dataset

# Train
python -m src.models.train --config configs/train_config.yaml

# Evaluate
python -m src.models.evaluate --model models/best_model.pt

# Run API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Generate Grad-CAM visualizations
python -m src.visualization.run_gradcam
```

---

## Future Improvements

1. **Ensemble Methods**: Combine EfficientNet with DenseNet for improved robustness
2. **Test-Time Augmentation**: Average predictions over augmented inputs
3. **Uncertainty Quantification**: Monte Carlo dropout for confidence calibration
4. **Active Learning**: Identify uncertain samples for expert review
5. **DICOM Support**: Native medical imaging format handling

---

## Conclusion

This project demonstrates a complete ML system that goes beyond model training to include:
- Proper data validation and handling of class imbalance
- Interpretability through Grad-CAM visualizations
- Production-ready API deployment with Docker support
- Comprehensive documentation and reproducibility

The high AUC scores across all classes indicate the model learns meaningful diagnostic features, while the explainability visualizations show it attends to clinically relevant regions of the X-rays.

---

*Built with PyTorch, timm, FastAPI, and best MLOps practices*

