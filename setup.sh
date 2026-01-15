#!/bin/bash

# Setup script for Image Classification Project

echo "================================================"
echo "Image Classification Project - Setup"
echo "================================================"

# Check Python version
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/saved_models
mkdir -p models/checkpoints
mkdir -p logs

echo ""
echo "================================================"
echo "Setup completed successfully!"
echo "================================================"
echo ""
echo "To activate the environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  venv\\Scripts\\activate"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "To train a model:"
echo "  python train.py --data_dir data/raw"
echo ""
echo "To make predictions:"
echo "  python predict.py --model_path models/saved_models/best_model.pth --image_path path/to/image.jpg --num_classes 10"
echo ""
