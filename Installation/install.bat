@echo off
REM Podcast Insight Engine - Windows Installation Script

echo ==========================================
echo Podcast Insight Engine - Installation
echo ==========================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo Python detected
echo.

REM Check pip
echo Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not installed
    pause
    exit /b 1
)
echo pip is available
echo.

REM Create virtual environment
set /p create_venv="Create virtual environment? (recommended) [y/N]: "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Virtual environment created and activated
    echo.
)

REM Install dependencies
echo Installing dependencies...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies
    pause
    exit /b 1
)
echo Dependencies installed
echo.

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True)"
echo NLTK data downloaded
echo.

REM Check for dataset
echo Checking for dataset...
if exist "this_american_life_transcripts.csv" (
    echo Dataset found
    set use_sample=false
) else (
    echo Dataset not found
    set /p create_sample="Create sample data for testing? [Y/n]: "
    if /i not "%create_sample%"=="n" (
        echo Creating sample data...
        python create_sample_data.py
        set use_sample=true
        echo Sample data created
    ) else (
        echo Please download the dataset and name it 'this_american_life_transcripts.csv'
        pause
        exit /b 1
    )
)
echo.

REM Run setup
echo Setting up the system...
if "%use_sample%"=="true" (
    echo Running in sample mode...
    python setup_data.py --sample
) else (
    set /p sample_mode="Use sample mode for faster setup? [y/N]: "
    if /i "%sample_mode%"=="y" (
        python setup_data.py --sample
    ) else (
        echo Running full setup (this will take 30-60 minutes)...
        python setup_data.py
    )
)
echo.

REM Run tests
echo Running system tests...
python test_system.py
echo.

REM Final instructions
echo ==========================================
echo Installation Complete!
echo ==========================================
echo.
echo To start the application:
echo   streamlit run app.py
echo.
if /i "%create_venv%"=="y" (
    echo Note: Remember to activate the virtual environment:
    echo   venv\Scripts\activate.bat
    echo.
)
echo For help, see README.md or QUICKSTART.md
echo ==========================================
pause