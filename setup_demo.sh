#!/bin/bash
# THRML MNIST Diffusion Demo - Quick Setup Script

set -e  # Exit on error

echo "=========================================="
echo "THRML MNIST Diffusion Demo Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo ""
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install THRML
echo ""
echo "Installing THRML..."
pip install -e .

# Install demo requirements
echo ""
echo "Installing demo requirements..."
pip install -r demo_requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the demo:"
echo "  1. Activate the virtual environment:"
echo "       source venv/bin/activate"
echo ""
echo "  2. Run the demo:"
echo "       python demo_mnist_diffusion.py --digit 1 --epochs 3"
echo ""
echo "  Or for a quick test:"
echo "       python demo_mnist_diffusion.py --digit 1 --epochs 1"
echo ""
echo "=========================================="
