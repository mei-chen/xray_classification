"""Setup script for the chest X-ray classification package."""

from setuptools import find_packages, setup

setup(
    name="xray_classification",
    version="0.1.0",
    description="Chest X-Ray Classification for Normal, Pneumonia, and Tuberculosis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
    },
)

