#!/bin/bash
# Podcast Insight Engine - Installation Script

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Podcast Insight Engine - Installation"
echo "=========================================="
echo ""

# Function for error messages
error_exit() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

# Function for success messages
success_msg() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function for warning messages
warning_msg() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if running on macOS or Linux
OS=$(uname -s)
echo "Detected OS: $OS"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    error_exit "python3 is not installed. Please install Python 3.8 or higher."
fi

python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+' || python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

# Better version comparison (works without bc)
version_compare() {
    python3 -c "import sys; sys.exit(0 if tuple(map(int, '$1'.split('.'))) >= tuple(map(int, '$2'.split('.'))) else 1)"
}

if ! version_compare "$python_version" "$required_version"; then
    error_exit "Python $required_version or higher is required. Current version: Python $python_version"
fi

success_msg "Python $python_version detected"
echo ""

# Check if pip is installed
echo "Checking pip..."
if ! command -v pip3 &> /dev/null; then
    error_exit "pip3 is not installed. Please install pip3."
fi

success_msg "pip3 is available"
echo ""

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    error_exit "requirements.txt not found in current directory"
fi

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-Y}  # Default to Y

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d "venv" ]; then
        warning_msg "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    python3 -m venv venv || error_exit "Failed to create virtual environment"
    
    # Activate virtual environment (different for different shells)
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        error_exit "Virtual environment activation script not found"
    fi
    
    success_msg "Virtual environment created and activated"
    echo ""
fi

# Upgrade pip to avoid warnings
echo "Upgrading pip..."
pip3 install --upgrade pip --quiet || warning_msg "Could not upgrade pip"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."

# Install with better error handling
if pip3 install -r requirements.txt; then
    success_msg "Dependencies installed"
else
    error_exit "Failed to install dependencies. Check requirements.txt and your internet connection."
fi
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 << EOF || error_exit "Failed to download NLTK data"
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    exit(1)
EOF
success_msg "NLTK data downloaded"
echo ""

# Check for dataset
echo "Checking for dataset..."
use_sample=false

if [ -f "this_american_life_transcripts.csv" ]; then
    success_msg "Dataset found: this_american_life_transcripts.csv"
    use_sample=false
else
    warning_msg "Dataset not found: this_american_life_transcripts.csv"
    read -p "Create sample data for testing? [Y/n]: " create_sample
    create_sample=${create_sample:-Y}
    
    if [[ ! $create_sample =~ ^[Nn]$ ]]; then
        if [ -f "create_sample_data.py" ]; then
            echo "Creating sample data..."
            python3 create_sample_data.py || error_exit "Failed to create sample data"
            use_sample=true
            success_msg "Sample data created"
        else
            error_exit "create_sample_data.py not found"
        fi
    else
        error_exit "Please download the dataset and name it 'this_american_life_transcripts.csv'"
    fi
fi
echo ""

# Run setup
echo "Setting up the system..."
if [ ! -f "setup_data.py" ]; then
    error_exit "setup_data.py not found"
fi

if [ "$use_sample" = true ]; then
    echo "Running in sample mode (faster)..."
    python3 setup_data.py --sample || error_exit "Setup failed"
else
    read -p "Use sample mode for faster setup? [Y/n]: " sample_mode
    sample_mode=${sample_mode:-Y}
    
    if [[ $sample_mode =~ ^[Yy]$ ]]; then
        python3 setup_data.py --sample || error_exit "Setup failed"
    else
        warning_msg "Running full setup (this will take 30-60 minutes)..."
        python3 setup_data.py || error_exit "Setup failed"
    fi
fi
echo ""

# Run tests
if [ -f "test_system.py" ]; then
    echo "Running system tests..."
    if python3 test_system.py; then
        success_msg "All tests passed"
    else
        warning_msg "Some tests failed. The application may still work."
    fi
else
    warning_msg "test_system.py not found, skipping tests"
fi
echo ""

# Final instructions
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo ""

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "To use the application, first activate the virtual environment:"
    echo "  source venv/bin/activate"
    echo ""
fi

echo "The Podcast Insight Engine is now ready to use!"
echo ""

if [ -f "README.md" ]; then
    echo "For help, see README.md"
fi

if [ -f "QUICKSTART.md" ]; then
    echo "Quick start guide: QUICKSTART.md"
fi

echo "=========================================="