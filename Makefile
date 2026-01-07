.PHONY: help setup data train evaluate deploy clean test lint format

# Default target
help:
	@echo "Chest X-Ray Classification Project"
	@echo ""
	@echo "Usage:"
	@echo "  make setup      - Create virtual environment and install dependencies"
	@echo "  make data       - Download and preprocess the dataset"
	@echo "  make train      - Train the model"
	@echo "  make evaluate   - Evaluate the trained model"
	@echo "  make deploy     - Run the FastAPI server"
	@echo "  make docker     - Build and run Docker container"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code"
	@echo "  make clean      - Remove generated files"
	@echo ""

# Environment setup
setup:
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -e .
	@echo "Setup complete! Activate with: source venv/bin/activate"

# Data pipeline
data:
	python -m src.data.download_dataset
	python -m src.data.make_dataset

# Data validation
validate:
	python -m src.data.validate_data --data-dir data/raw/chest-xray-dataset

validate-fast:
	python -m src.data.validate_data --data-dir data/raw/chest-xray-dataset --no-duplicates

# Model training
train:
	python -m src.models.train --config configs/train_config.yaml

# Model evaluation
evaluate:
	python -m src.models.evaluate --config configs/train_config.yaml

# Explainability (Grad-CAM)
gradcam:
	python -m src.visualization.run_gradcam --n-per-class 3

# Deploy API
deploy:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t xray-classifier:latest .

docker-run:
	docker run -p 8000:8000 --rm xray-classifier:latest

docker: docker-build docker-run

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Code quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf venv build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Remove data and models (use with caution)
clean-all: clean
	rm -rf data/raw/* data/interim/* data/processed/*
	rm -rf models/*
	rm -rf reports/figures/*

