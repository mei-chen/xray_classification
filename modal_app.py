"""
Modal deployment for Chest X-Ray Classification API.

Deploy with:
    modal deploy modal_app.py

Run locally (for testing):
    modal serve modal_app.py
"""

import io
import modal

# Define the Modal app
app = modal.App("xray-classifier")

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "python-multipart>=0.0.6",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("models", "/root/models")
)

# Create a volume to persist the model (optional, for faster cold starts)
model_volume = modal.Volume.from_name("xray-model-volume", create_if_missing=True)


@app.cls(
    image=image,
    gpu=None,  # CPU is fine for inference, set to "T4" for GPU
    scaledown_window=300,  # Keep warm for 5 minutes
)
class XRayClassifier:
    """Modal class for X-Ray classification inference."""
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        import sys
        sys.path.insert(0, "/root")
        
        import torch
        from pathlib import Path
        
        from src.config import get_device, load_config
        from src.data.transforms import get_inference_transforms
        from src.models.architecture import load_model
        from src.visualization.visualize import GradCAM
        
        # Load config
        config_path = Path("/root/configs/deploy_config.yaml")
        if config_path.exists():
            self.config = load_config(config_path)
        else:
            self.config = {
                "inference": {
                    "model_path": "/root/models/best_model.pt",
                    "image_size": 224,
                    "classes": ["Normal", "Pneumonia", "Tuberculosis"]
                }
            }
        
        inference_config = self.config.get("inference", {})
        
        # Setup device (CPU for Modal, unless GPU specified)
        self.device = get_device("cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        model_path = inference_config.get("model_path", "/root/models/best_model.pt")
        self.model = load_model(
            model_path,
            device=self.device,
            architecture=self.config.get("model", {}).get("architecture", "efficientnet_b0"),
            num_classes=3,
        )
        self.model.eval()
        print("Model loaded successfully!")
        
        # Setup transforms
        self.image_size = inference_config.get("image_size", 224)
        self.transform = get_inference_transforms(image_size=self.image_size)
        
        # Class names
        self.class_names = inference_config.get("classes", ["Normal", "Pneumonia", "Tuberculosis"])
        
        # Setup Grad-CAM
        target_layer = None
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer:
            self.gradcam = GradCAM(self.model, target_layer)
        else:
            self.gradcam = None
        
        print("XRay Classifier ready!")
    
    def _preprocess(self, image_bytes: bytes):
        """Preprocess image bytes to tensor."""
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, tensor
    
    @modal.method()
    def predict(self, image_bytes: bytes) -> dict:
        """Classify a chest X-ray image."""
        import time
        import torch
        
        start_time = time.time()
        
        _, tensor = self._preprocess(image_bytes)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(dim=1).item()
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
            "inference_time_ms": round(inference_time, 2),
        }
    
    @modal.method()
    def predict_with_gradcam(self, image_bytes: bytes) -> dict:
        """Classify with Grad-CAM explanation."""
        import base64
        import time
        import numpy as np
        import torch
        from PIL import Image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if self.gradcam is None:
            return {"error": "Grad-CAM not available"}
        
        start_time = time.time()
        
        original_image, tensor = self._preprocess(image_bytes)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(dim=1).item()
        
        # Generate Grad-CAM
        cam = self.gradcam.generate(tensor, pred_idx)
        
        # Resize CAM
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size),
                Image.BILINEAR,
            )
        ) / 255.0
        
        # Prepare images
        img_resized = original_image.resize((self.image_size, self.image_size))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Create heatmap
        heatmap_colored = plt.cm.jet(cam_resized)[:, :, :3]
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
        
        # Create overlay
        overlay = 0.5 * img_array + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        
        # Convert to base64
        def to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
            "inference_time_ms": round(inference_time, 2),
            "gradcam_image": to_base64(overlay_img),
            "heatmap_image": to_base64(heatmap_img),
        }


# FastAPI web endpoints
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

web_app = FastAPI(
    title="Chest X-Ray Classification API",
    description="Classify chest X-rays into Normal, Pneumonia, or Tuberculosis",
    version="1.0.0",
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """API information."""
    return {
        "service": "Chest X-Ray Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "explain": "/predict/explain",
            "docs": "/docs",
        }
    }


@web_app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "model": "loaded"}


@web_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify a chest X-ray image."""
    if file.content_type not in ["image/jpeg", "image/png", "image/gif", "image/bmp"]:
        raise HTTPException(400, f"Invalid file type: {file.content_type}")
    
    image_bytes = await file.read()
    classifier = XRayClassifier()
    result = classifier.predict.remote(image_bytes)
    return result


@web_app.post("/predict/explain")
async def predict_explain(file: UploadFile = File(...)):
    """Classify with Grad-CAM explanation."""
    if file.content_type not in ["image/jpeg", "image/png", "image/gif", "image/bmp"]:
        raise HTTPException(400, f"Invalid file type: {file.content_type}")
    
    image_bytes = await file.read()
    classifier = XRayClassifier()
    result = classifier.predict_with_gradcam.remote(image_bytes)
    return result


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Expose FastAPI app via Modal."""
    return web_app


# CLI entry point for direct method calls
@app.local_entrypoint()
def main():
    """Test the classifier locally."""
    from pathlib import Path
    
    # Find a test image
    test_images = list(Path("data/raw/chest-xray-dataset/test/pneumonia").glob("*.jpg"))[:1]
    
    if not test_images:
        print("No test images found. Deploy with: modal deploy modal_app.py")
        return
    
    classifier = XRayClassifier()
    
    for img_path in test_images:
        print(f"\nClassifying: {img_path.name}")
        with open(img_path, "rb") as f:
            result = classifier.predict.remote(f.read())
        
        print(f"  Prediction: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Inference time: {result['inference_time_ms']:.1f}ms")

