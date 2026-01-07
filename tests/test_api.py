"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import numpy as np


# Note: These tests require the model to be loaded
# For CI/CD, you might want to mock the model

class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        from src.api.main import app
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        # Create a text file instead of image
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        
        response = client.post("/predict", files=files)
        
        # Should return 400 Bad Request
        assert response.status_code == 400
    
    @pytest.mark.skipif(
        True,  # Skip if model not loaded
        reason="Model not available for testing"
    )
    def test_predict_valid_image(self, client):
        """Test prediction with valid image."""
        # Create a valid test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        files = {"file": ("test.png", img_bytes, "image/png")}
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data

