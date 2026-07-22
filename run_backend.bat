@echo off
echo ============================================
echo   Cancer Prediction App - Backend
echo ============================================
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install from https://python.org  (check "Add to PATH")
    pause & exit /b
)
if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)
echo [2/3] Installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt --quiet
echo [3/3] Starting Flask server...
echo.
echo  Backend : http://localhost:5000
echo  Frontend: open index.html in your browser
echo  Press Ctrl+C to stop
echo.
python app.py
pause
