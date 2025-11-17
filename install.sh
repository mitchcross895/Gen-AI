#!/bin/bash

# Podcast Insight Engine - Installation Script
# This script automates the setup process

set -e  # Exit on error

echo "=========================================="
echo "Podcast Insight Engine - Installation"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
required_version="3.8"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "❌ Error: Python $required_version or higher is required"
    echo "   Current version: Python $python_version"
    exit 1
fi
echo "✓ Python $python_version detected"
echo ""

# Check if pip is installed
echo "Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed"
    exit 1
fi
echo "✓ pip3 is available"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."
pip3 install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ NLTK data downloaded"
echo ""

# Check for dataset
echo "Checking for dataset..."
if [ -f "this_american_life_transcripts.csv" ]; then
    echo "✓ Dataset found"
    use_sample=false
else
    echo "⚠ Dataset not found"
    read -p "Create sample data for testing? [Y/n]: " create_sample
    if [[ ! $create_sample =~ ^[Nn]$ ]]; then
        echo "Creating sample data..."
        python3 create_sample_data.py
        use_sample=true
        echo "✓ Sample data created"
    else
        echo "Please download the dataset and name it 'this_american_life_transcripts.csv'"
        exit 1
    fi
fi
echo ""

# Run setup
echo "Setting up the system..."
if [ "$use_sample" = true ]; then
    echo "Running in sample mode (faster)..."
    python3 setup_data.py --sample
else
    read -p "Use sample mode for faster setup? [y/N]: " sample_mode
    if [[ $sample_mode =~ ^[Yy]$ ]]; then
        python3 setup_data.py --sample
    else
        echo "Running full setup (this will take 30-60 minutes)..."
        python3 setup_data.py
    fi
fi
echo ""

# Run tests
echo "Running system tests..."
python3 test_system.py
echo ""

# Final instructions
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo "  streamlit run app.py"
echo ""
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Note: Remember to activate the virtual environment:"
    echo "  source venv/bin/activate"
    echo ""
fi
echo "For help, see README.md or QUICKSTART.md"
echo "=========================================="