# Quick Start Guide

## 1. Setup Environment

```bash
cd xray_classification
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then visit: **http://localhost:8000/docs** for interactive API documentation.

## 3. Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/xray.jpg"
```

## 4. Get Explainability (Grad-CAM)

```bash
curl -X POST "http://localhost:8000/predict/explain" \
  -H "accept: application/json" \
  -F "file=@path/to/xray.jpg"
```

Returns prediction + base64-encoded attention heatmap.

## 5. Docker Deployment

```bash
docker build -t xray-classifier .
docker run -p 8000:8000 xray-classifier
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/model/info` | GET | Model details |
| `/predict` | POST | Classify image |
| `/predict/batch` | POST | Classify multiple |
| `/predict/explain` | POST | Classify + Grad-CAM |

---

## Sample Test Images

Located at: `data/raw/chest-xray-dataset/test/`
- `normal/normal-1.jpg`
- `pneumonia/pneumonia-1.jpg`
- `tuberculosis/tuberculosis-1.jpg`

