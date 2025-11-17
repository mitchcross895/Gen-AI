# Installation Guide

Complete guide for setting up the Podcast Insight Engine with two options: quick start with your own data, or full setup with the This American Life dataset from Kaggle.

---

## Prerequisites

- Python 3.8 or higher
- 2GB+ RAM (4GB+ recommended for full dataset)
- 5GB+ free disk space (for full dataset with embeddings)
- Internet connection

---

## Option 1: Quick Setup (Your Own Dataset)

**Time:** ~5-10 minutes  
**Best for:** Testing with your own podcast transcripts

### Step 1: Clone/Download the Project

```bash
# If using git
git clone <your-repo-url>
cd podcast-insight-engine

# Or download and extract the ZIP file
```

### Step 2: Run Installation Script

#### Windows
```bash
install.bat
```

This will:
- Create virtual environment
- Install Python dependencies
- Download spaCy model
- Verify installation

#### Linux/Mac
```bash
chmod +x install.sh
./install.sh
```

### Step 3: Verify Installation

```bash
# Activate virtual environment first
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Run test
python test_system.py
```

You should see: `ALL TESTS PASSED! ✓`

### Step 4: Launch Application

```bash
streamlit run main.py
```

Upload your CSV file and start exploring!

---

## Option 2: Full Setup (This American Life Dataset)

**Time:** ~30-60 minutes (mostly automated)  
**Best for:** Full experience with real podcast data (~700 episodes)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 2: Setup Kaggle API Access

#### 2a. Create Kaggle Account
1. Go to https://www.kaggle.com
2. Sign up or sign in

#### 2b. Get API Token
1. Click your profile icon (top right)
2. Select "Settings"
3. Scroll to "API" section
4. Click "Create New API Token"
5. This downloads `kaggle.json` to your Downloads folder

#### 2c. Install API Token

**Linux/Mac:**
```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Move the token file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Secure the file
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```bash
# Create kaggle directory
mkdir %USERPROFILE%\.kaggle

# Move the token file (adjust path if needed)
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

#### 2d. Verify Kaggle Setup (Recommended)

```bash
python setup_kaggle.py
```

This interactive script will:
- Check if credentials are in the right place
- Verify they're valid
- Test the connection
- Help fix any issues

You should see: `✓ KAGGLE API READY!`

### Step 3: Download and Process Dataset

```bash
python setup_data.py
```

This will automatically:
1. Download ~700 episodes from Kaggle (~10 min)
2. Clean and process transcripts (~5 min)
3. Generate semantic embeddings (~30-45 min)
4. Create summary statistics (~2 min)
5. Run test searches to verify

**Options:**
```bash
# Use smaller batches (if low memory)
python setup_data.py --batch-size 16

# Skip embeddings for now (download and process only)
python setup_data.py --skip-embeddings

# Force fresh download
python setup_data.py --force-redownload
```

### Step 4: Launch Application

```bash
streamlit run main.py
```

The app opens with the full dataset loaded and ready!

---

## Manual Installation (All Steps)

If the automated scripts don't work, follow these manual steps:

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies
```bash
pip install pandas numpy streamlit
pip install sentence-transformers transformers torch
pip install spacy scikit-learn tqdm python-dotenv
pip install kagglehub  # Only if using Kaggle dataset
```

### 5. Download Language Model
```bash
python -m spacy download en_core_web_sm
```

### 6. Verify Installation
```bash
python test_system.py
```

---

## Troubleshooting

### Issue: Python version too old
**Error:** `Python 3.8 or higher required`

**Solution:**
```bash
# Check your Python version
python --version

# Install Python 3.8+ from python.org
# Then retry installation
```

### Issue: pip not found
**Error:** `pip: command not found`

**Solution:**
```bash
# Try python -m pip instead
python -m pip install -r requirements.txt
```

### Issue: Permission denied (Linux/Mac)
**Error:** `Permission denied`

**Solution:**
```bash
# Don't use sudo, use virtual environment instead
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Kaggle credentials not found
**Error:** `Could not find kaggle.json`

**Solution:**
```bash
# Run the setup helper
python setup_kaggle.py

# Or manually:
# 1. Download from kaggle.com/account
# 2. Place at ~/.kaggle/kaggle.json (Linux/Mac)
#    or C:\Users\<You>\.kaggle\kaggle.json (Windows)
# 3. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac only)
```

### Issue: Out of memory during setup
**Error:** `Out of memory` or process crashes

**Solution:**
```bash
# Use smaller batch size
python setup_data.py --batch-size 8

# Or skip embeddings and generate later
python setup_data.py --skip-embeddings
```

### Issue: torch/CUDA errors
**Error:** `No CUDA-capable device detected`

**Solution:**
This is fine! The system works on CPU. If you want GPU acceleration:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: spaCy model fails to download
**Error:** `Can't find model 'en_core_web_sm'`

**Solution:**
```bash
# Try with full path
python -m spacy download en_core_web_sm

# Or download manually
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
```

### Issue: Streamlit not found
**Error:** `streamlit: command not found`

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install streamlit
pip install streamlit

# Run with python -m
python -m streamlit run main.py
```

---

## Verifying Installation

After installation, verify everything works:

```bash
# 1. Check Python packages
pip list

# 2. Run test script
python test_system.py

# 3. Start the app
streamlit run main.py

# 4. (Optional) Test CLI
python run.py --help
```

---

## System Requirements

### Minimum
- Python 3.8+
- 2GB RAM
- 1GB free disk space
- CPU only

### Recommended
- Python 3.9+
- 8GB RAM
- 10GB free disk space (for full dataset)
- GPU with CUDA (optional, speeds up embeddings)

### For Full Dataset (Option 2)
- 4GB+ RAM
- 5GB+ free disk space
- 30-60 minutes processing time
- Stable internet connection

---

## Next Steps

After successful installation:

1. **Read the Quick Start Guide:** `QUICKSTART.md`
2. **Check the Documentation:** `README.md`
3. **Prepare your data:** See dataset format in README
4. **Start exploring!** Run `streamlit run main.py`

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review error messages carefully
3. Try manual installation steps
4. Verify all prerequisites are met
5. Check that virtual environment is activated

Common mistakes:
- ❌ Running without activating virtual environment
- ❌ Using Python 2.x instead of 3.x
- ❌ Missing Kaggle credentials for Option 2
- ❌ Insufficient disk space
- ❌ Firewall blocking downloads

---

## Uninstallation

To completely remove the installation:

```bash
# 1. Deactivate virtual environment
deactivate

# 2. Delete project directory
rm -rf podcast-insight-engine  # Linux/Mac
rmdir /s podcast-insight-engine  # Windows

# 3. (Optional) Remove Kaggle credentials
rm ~/.kaggle/kaggle.json  # Linux/Mac
del %USERPROFILE%\.kaggle\kaggle.json  # Windows
```