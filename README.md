# Chest X-Ray Classification

> **[Live Demo](https://xray.agentnow.org/)** - Try the deployed model

A production-ready deep learning solution for classifying chest X-ray images into three categories:
- **Normal** - Healthy chest X-rays
- **Pneumonia** - X-rays showing pneumonia infection
- **Tuberculosis** - X-rays showing tuberculosis indicators

## 🎯 Project Goals

- Demonstrate expertise in data handling, modeling, evaluation, and deployment
- Build a reproducible, well-documented ML pipeline
- Create a deployable API service for real-time predictions

## 📁 Project Structure

```
├── configs/               <- Configuration files (hyperparameters, paths)
├── data/
│   ├── raw/              <- Original, immutable data dump
│   ├── interim/          <- Intermediate data transformations
│   ├── processed/        <- Final, canonical data sets for modeling
│   └── external/         <- Data from third party sources
├── docs/                 <- Documentation files
├── models/               <- Trained and serialized models
├── notebooks/            <- Jupyter notebooks for exploration
├── reports/
│   └── figures/          <- Generated graphics and figures
├── src/
│   ├── data/             <- Data loading and preprocessing
│   ├── features/         <- Feature engineering
│   ├── models/           <- Model architectures and training
│   ├── visualization/    <- Plotting utilities
│   └── api/              <- FastAPI deployment
└── tests/                <- Unit tests
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
python -m src.data.download_dataset
```

### Train Model

```bash
python -m src.models.train --config configs/train_config.yaml
```

### Run API Server

```bash
uvicorn src.api.main:app --reload
```

### Docker Deployment

```bash
docker build -t xray-classifier .
docker run -p 8000:8000 xray-classifier
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 77.1% |
| Macro AUC-ROC | 90.8% |
| Pneumonia AUC | 98.7% |
| Test Samples | 2,569 |

## 🔬 Methodology

### Data Pipeline
1. Download from Kaggle using `kagglehub`
2. Exploratory Data Analysis (class distribution, image statistics)
3. Data augmentation (rotation, flipping, brightness adjustment)
4. Train/validation/test split with stratification

### Model Architecture
- **Base**: EfficientNet-B0 (pretrained on ImageNet)
- **Head**: Global Average Pooling → Dropout → Dense(3, softmax)
- **Transfer Learning**: Progressive unfreezing strategy

### Training Strategy
- Mixed precision training (FP16) for efficiency
- Learning rate scheduling (Cosine Annealing with Warm Restarts)
- Early stopping with model checkpointing
- Class-weighted loss for imbalanced data

### Evaluation
- Confusion matrix and classification report
- ROC curves and AUC scores per class
- Grad-CAM visualizations for interpretability

## 🔧 Configuration

All hyperparameters are managed via YAML config files in `configs/`:

```yaml
# configs/train_config.yaml
model:
  architecture: efficientnet_b0
  pretrained: true
  dropout: 0.3

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  
data:
  image_size: 224
  augmentation: true
```

## 📝 Reproducibility

- All random seeds are fixed (Python, NumPy, PyTorch)
- Full experiment logging with training metrics
- Model versioning with metadata

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines first.

---

*Built with ❤️ using PyTorch, FastAPI, and best MLOps practices*

