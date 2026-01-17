"""
FastAPI application for Chest X-Ray Classification.

Provides REST API endpoints for:
- Health check
- Single image prediction
- Batch prediction
- Explainability (Grad-CAM)
- Model information

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.config import get_device, load_config
from src.data.transforms import get_inference_transforms
from src.models.architecture import load_model
from src.visualization.visualize import GradCAM, LayerCAM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and config
model = None
transform = None
device = None
class_names = None
config = None
gradcam_instance = None
layercam_instance = None
target_layer = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoints."""
    
    predicted_class: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(..., description="Probabilities for all classes")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    
    predictions: list[PredictionResponse]
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str
    model_loaded: bool
    device: str
    model_architecture: str | None


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    
    architecture: str
    num_classes: int
    class_names: list[str]
    input_size: int
    total_parameters: int
    device: str


class ExplainResponse(BaseModel):
    """Response model for explainability endpoint with Grad-CAM."""
    
    predicted_class: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(..., description="Probabilities for all classes")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    gradcam_image: str = Field(..., description="Base64-encoded Grad-CAM overlay image (PNG)")
    heatmap_image: str = Field(..., description="Base64-encoded heatmap image (PNG)")


class MultiCAMResponse(BaseModel):
    """Response model for multi-method CAM explainability."""
    
    predicted_class: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(..., description="Probabilities for all classes")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    gradcam_overlay: str = Field(..., description="Base64-encoded Grad-CAM overlay (PNG)")
    gradcam_heatmap: str = Field(..., description="Base64-encoded Grad-CAM heatmap (PNG)")
    layercam_overlay: str = Field(..., description="Base64-encoded Layer-CAM overlay (PNG)")
    layercam_heatmap: str = Field(..., description="Base64-encoded Layer-CAM heatmap (PNG)")


def load_model_on_startup():
    """Load the model during application startup."""
    global model, transform, device, class_names, config, gradcam_instance, layercam_instance, target_layer
    
    # Try to load deploy config, fall back to train config
    config_path = Path("configs/deploy_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/train_config.yaml")
    
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {
            "inference": {
                "model_path": "models/best_model.pt",
                "device": "auto",
                "image_size": 224,
            }
        }
    
    inference_config = config.get("inference", config.get("data", {}))
    
    # Get device
    device = get_device(inference_config.get("device", "auto"))
    logger.info(f"Using device: {device}")
    
    # Load model
    model_path = inference_config.get("model_path", "models/best_model.pt")
    
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}. API will return errors until model is available.")
        return
    
    try:
        model = load_model(
            model_path,
            device=device,
            architecture=config.get("model", {}).get("architecture", "efficientnet_b0"),
            num_classes=config.get("data", {}).get("num_classes", 3),
        )
        logger.info(f"Model loaded from: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Setup CAM instances for explainability
    try:
        target_layer = None
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is not None:
            gradcam_instance = GradCAM(model, target_layer)
            layercam_instance = LayerCAM(model, target_layer)
            logger.info("Grad-CAM and Layer-CAM initialized for explainability")
    except Exception as e:
        logger.warning(f"Could not initialize CAM instances: {e}")
    
    # Setup transforms
    image_size = inference_config.get("image_size", 224)
    transform = get_inference_transforms(image_size=image_size)
    
    # Class names
    class_names = inference_config.get(
        "classes",
        config.get("data", {}).get("classes", ["Normal", "Pneumonia", "Tuberculosis"])
    )
    
    logger.info(f"Class names: {class_names}")
    logger.info("Model ready for inference!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up Chest X-Ray Classification API...")
    load_model_on_startup()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Chest X-Ray Classification API",
    description="""
    API for classifying chest X-ray images into three categories:
    - **Normal**: Healthy chest X-rays
    - **Pneumonia**: X-rays showing pneumonia infection
    - **Tuberculosis**: X-rays showing tuberculosis indicators
    
    ## Usage
    
    Upload a chest X-ray image to the `/predict` endpoint to get a classification.
    
    ## Explainability
    
    Use `/predict/explain` to get Grad-CAM visualizations showing which regions
    of the image influenced the model's decision.
    
    ## Model
    
    The model uses transfer learning with EfficientNet-B0 pretrained on ImageNet,
    fine-tuned on a dataset of chest X-ray images.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess an image for model inference.
    
    Args:
        image_bytes: Raw image bytes.
        
    Returns:
        Preprocessed image tensor.
        
    Raises:
        ValueError: If image cannot be processed.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        return tensor.to(device)
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")


def predict_single(image_tensor: torch.Tensor) -> tuple[str, float, dict[str, float], float]:
    """Make prediction on a single image.
    
    Args:
        image_tensor: Preprocessed image tensor.
        
    Returns:
        Tuple of (predicted_class, confidence, probabilities, inference_time).
    """
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = outputs.argmax(dim=1).item()
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    predicted_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    probabilities = {
        name: probs[i].item()
        for i, name in enumerate(class_names)
    }
    
    return predicted_class, confidence, probabilities, inference_time


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Chest X-Ray Classification API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "explain": "/predict/explain",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_architecture=config.get("model", {}).get("architecture") if config else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check the model path.",
        )
    
    inference_config = config.get("inference", config.get("data", {}))
    
    return ModelInfoResponse(
        architecture=config.get("model", {}).get("architecture", "efficientnet_b0"),
        num_classes=len(class_names),
        class_names=class_names,
        input_size=inference_config.get("image_size", 224),
        total_parameters=model.get_total_params(),
        device=str(device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(description="Chest X-ray image file")]
):
    """
    Classify a chest X-ray image.
    
    Upload a chest X-ray image (JPEG, PNG) to receive a classification
    and confidence scores for each class.
    
    **Supported formats**: JPEG, PNG, GIF, BMP
    
    **Returns**: Predicted class, confidence score, and probabilities for all classes.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists.",
        )
    
    # Validate file type
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif", "image/bmp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Supported: JPEG, PNG, GIF, BMP",
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # Get prediction
        predicted_class, confidence, probabilities, inference_time = predict_single(image_tensor)
        
        logger.info(
            f"Prediction: {predicted_class} ({confidence:.2%}) - "
            f"Inference time: {inference_time:.1f}ms"
        )
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=round(inference_time, 2),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: Annotated[list[UploadFile], File(description="List of chest X-ray images")]
):
    """
    Classify multiple chest X-ray images.
    
    Upload multiple chest X-ray images to receive classifications
    for all of them in a single request.
    
    **Maximum**: 10 images per request
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch request.",
        )
    
    predictions = []
    total_start = time.time()
    
    for file in files:
        try:
            image_bytes = await file.read()
            image_tensor = preprocess_image(image_bytes)
            predicted_class, confidence, probabilities, inference_time = predict_single(image_tensor)
            
            predictions.append(PredictionResponse(
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                inference_time_ms=round(inference_time, 2),
            ))
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing {file.filename}: {e}",
            )
    
    total_time = (time.time() - total_start) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_inference_time_ms=round(total_time, 2),
    )


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_with_explanation(
    file: Annotated[UploadFile, File(description="Chest X-ray image file")]
):
    """
    Classify a chest X-ray image with Grad-CAM explainability.
    
    Returns the prediction along with Grad-CAM visualization showing
    which regions of the image most influenced the model's decision.
    
    **Returns**:
    - Predicted class and confidence scores
    - `gradcam_image`: Base64-encoded PNG showing the overlay of attention on the original image
    - `heatmap_image`: Base64-encoded PNG showing just the attention heatmap
    
    **Note**: This endpoint is slower than `/predict` due to gradient computation.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )
    
    if gradcam_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Grad-CAM not initialized. Explainability is unavailable.",
        )
    
    # Validate file type
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif", "image/bmp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Supported: JPEG, PNG, GIF, BMP",
        )
    
    try:
        start_time = time.time()
        
        # Read and preprocess image
        image_bytes = await file.read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(dim=1).item()
        
        predicted_class = class_names[pred_idx]
        confidence = probs[pred_idx].item()
        probabilities = {
            name: probs[i].item()
            for i, name in enumerate(class_names)
        }
        
        # Generate Grad-CAM
        cam = gradcam_instance.generate(image_tensor, pred_idx)
        
        # Get image size from config
        inference_config = config.get("inference", config.get("data", {}))
        image_size = inference_config.get("image_size", 224)
        
        # Resize CAM to image size
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (image_size, image_size),
                Image.BILINEAR,
            )
        ) / 255.0
        
        # Prepare original image for overlay
        img_resized = original_image.resize((image_size, image_size))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Create heatmap (just the CAM visualization)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Generate heatmap image
        heatmap_colored = plt.cm.jet(cam_resized)[:, :, :3]
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
        
        # Create overlay (original + heatmap)
        overlay = 0.5 * img_array + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        
        # Convert images to base64
        def image_to_base64(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        gradcam_base64 = image_to_base64(overlay_img)
        heatmap_base64 = image_to_base64(heatmap_img)
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Explain: {predicted_class} ({confidence:.2%}) - "
            f"Time: {inference_time:.1f}ms"
        )
        
        return ExplainResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=round(inference_time, 2),
            gradcam_image=gradcam_base64,
            heatmap_image=heatmap_base64,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post("/predict/explain/compare", response_model=MultiCAMResponse)
async def predict_with_multi_cam(
    file: Annotated[UploadFile, File(description="Chest X-ray image file")]
):
    """
    Classify a chest X-ray image with both Grad-CAM and Layer-CAM explainability.
    
    Compares two attention visualization methods:
    - **Grad-CAM**: Uses global average pooling of gradients (broader attention regions)
    - **Layer-CAM**: Uses element-wise spatial weighting (finer-grained attention)
    
    **Returns**:
    - Predicted class and confidence scores
    - Grad-CAM and Layer-CAM overlays and heatmaps as Base64-encoded PNGs
    
    **Note**: This endpoint is slower than single-method explainability due to multiple CAM computations.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )
    
    if gradcam_instance is None or layercam_instance is None:
        raise HTTPException(
            status_code=503,
            detail="CAM instances not initialized. Explainability is unavailable.",
        )
    
    # Validate file type
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif", "image/bmp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Supported: JPEG, PNG, GIF, BMP",
        )
    
    try:
        start_time = time.time()
        
        # Read and preprocess image
        image_bytes = await file.read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(dim=1).item()
        
        predicted_class = class_names[pred_idx]
        confidence = probs[pred_idx].item()
        probabilities = {
            name: probs[i].item()
            for i, name in enumerate(class_names)
        }
        
        # Get image size from config
        inference_config = config.get("inference", config.get("data", {}))
        image_size = inference_config.get("image_size", 224)
        
        # Prepare original image for overlay
        img_resized = original_image.resize((image_size, image_size))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        def process_cam(cam_instance, name: str):
            """Generate CAM and return overlay + heatmap images."""
            cam = cam_instance.generate(image_tensor, pred_idx)
            
            # Resize CAM to image size
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (image_size, image_size),
                    Image.BILINEAR,
                )
            ) / 255.0
            
            # Generate heatmap image
            heatmap_colored = plt.cm.jet(cam_resized)[:, :, :3]
            heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
            
            # Create overlay (original + heatmap)
            overlay = 0.5 * img_array + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
            
            return overlay_img, heatmap_img
        
        # Generate both CAMs
        gradcam_overlay, gradcam_heatmap = process_cam(gradcam_instance, "Grad-CAM")
        layercam_overlay, layercam_heatmap = process_cam(layercam_instance, "Layer-CAM")
        
        # Convert images to base64
        def image_to_base64(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Multi-CAM Explain: {predicted_class} ({confidence:.2%}) - "
            f"Time: {inference_time:.1f}ms"
        )
        
        return MultiCAMResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=round(inference_time, 2),
            gradcam_overlay=image_to_base64(gradcam_overlay),
            gradcam_heatmap=image_to_base64(gradcam_heatmap),
            layercam_overlay=image_to_base64(layercam_overlay),
            layercam_heatmap=image_to_base64(layercam_heatmap),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Multi-CAM explain error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

