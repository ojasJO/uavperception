@echo off
rem AeroTrack Research Studio launcher (Windows). Uses the project venv; never a system Python.
rem   studio.bat               -> http://localhost:8501
rem   studio.bat 8600          -> another port
rem Stop: Ctrl+C in this window (or close it).
setlocal
cd /d %~dp0
if not exist venv\Scripts\python.exe (
    echo [ERROR] venv\Scripts\python.exe not found. See docs\workspace\OPERATING_GUIDE.md section 1.
    exit /b 1
)
set PORT=%1
if "%PORT%"=="" set PORT=8501
echo [studio] checking the environment...
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from aerolab.health import environment as e; d = e(); print('[studio] python', d['python'], '| torch', d['packages'].get('torch'), '| CUDA', d.get('cuda_available'), '|', d.get('gpu') or 'CPU only', '| disk free', d['disk_free_gb'], 'GB')"
if not exist results\registry\models.json (
    echo [studio] registry missing: building it once...
    venv\Scripts\python.exe scripts\build_workspace_registry.py || exit /b 1
)
echo [studio] starting on http://localhost:%PORT%  (Ctrl+C to stop)
rem --server.address localhost: the studio is a local tool; it is not exposed on the network
venv\Scripts\python.exe -m streamlit run app\app.py --server.address localhost --server.port %PORT% --server.headless true --browser.gatherUsageStats false --client.toolbarMode minimal
