@echo off
rem ============================================================================
rem  DRONE-Track / AeroTrack-Net  --  ONE-CLICK DEMONSTRATION
rem
rem  Double-click this file. Nothing to install, nothing to type, no arguments.
rem  Press A for the full walkthrough (this is the one to screen-record), or
rem  pick a single section by number.
rem
rem  Every section runs read-only: it reads the datasets, the trained weights and
rem  the generated reports already in this folder. No section trains a model, and
rem  no section touches a TEST split.
rem ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title DRONE-Track -- one-click demonstration
rem a tall scrollback so the reports below can be scrolled back through; harmless if refused
mode con: cols=120 lines=3000 >nul 2>&1

set "PY=venv\Scripts\python.exe"
if not exist "%PY%" (
    echo.
    echo  [ERROR] %CD%\venv\Scripts\python.exe not found.
    echo          The project virtual environment is missing. Create it with:
    echo.
    echo             py -3.11 -m venv venv
    echo             venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    echo          Full instructions: docs\workspace\OPERATING_GUIDE.md section 1.
    echo.
    pause
    exit /b 1
)

:MENU
cls
echo ============================================================================
echo   DRONE-Track / AeroTrack-Net  --  micro-UAV detection and tracking
echo   in thermal infrared and visible video
echo ============================================================================
echo.
echo   Project folder : %CD%
echo.
echo   1   Environment and GPU check            -- what this machine is running
echo   2   Dataset directory information        -- where every dataset lives
echo   3   Trained models and checkpoints       -- what was trained, and its SHA-256
echo   4   All results                          -- every published number, one document
echo   5   LIVE DETECTION on real sequences     -- opens a video window
echo   6   AeroTrack Flight Deck                -- the full research laboratory, in a browser
echo   7   Run the test suite                   -- 122 tests, proves the code works
echo.
echo   A   RUN EVERYTHING IN ORDER  (1-2-3-4-5-6)   <-- record this one
echo   Q   Quit
echo.
set "CHOICE="
set /p "CHOICE=  Your choice: "
if /i "%CHOICE%"=="1" goto S1
if /i "%CHOICE%"=="2" goto S2
if /i "%CHOICE%"=="3" goto S3
if /i "%CHOICE%"=="4" goto S4
if /i "%CHOICE%"=="5" goto S5
if /i "%CHOICE%"=="6" goto S6
if /i "%CHOICE%"=="7" goto S7
if /i "%CHOICE%"=="A" goto ALL
if /i "%CHOICE%"=="Q" exit /b 0
goto MENU

rem ---------------------------------------------------------------------------
:ALL
set "RUNALL=1"
:S1
cls
call :BANNER "1 / 6   ENVIRONMENT AND GPU CHECK"
echo   Reading the Python, PyTorch, CUDA and GPU this machine will actually use.
echo.
"%PY%" tools\show_env.py
echo.
call :NEXT
if not defined RUNALL goto MENU

:S2
cls
call :BANNER "2 / 6   DATASET DIRECTORY INFORMATION"
echo   Every path below is read from the generated registry and then checked
echo   against this disk, so [ok] means the directory is really there.
echo.
"%PY%" tools\show_datasets.py --tree
echo.
call :NEXT
if not defined RUNALL goto MENU

:S3
cls
call :BANNER "3 / 6   TRAINED MODELS AND CHECKPOINTS"
echo   Every checkpoint is identified by its SHA-256, so the weights behind a
echo   published number can be told apart from any other copy.
echo.
"%PY%" tools\show_models.py
echo.
call :NEXT
if not defined RUNALL goto MENU

:S4
cls
call :BANNER "4 / 6   ALL RESULTS"
echo   SUBMISSION\RESULTS_ALL.md holds every existing result in one document:
echo   the CST benchmark headline, the Anti-UAV 2021 reproduction, the detector
echo   ablation, trajectory forecasting, deployment latency, and the components
echo   that were measured and retired.
echo.
echo   Opening it now. Press space to page through, Q to leave it.
echo.
pause
more "SUBMISSION\RESULTS_ALL.md"
echo.
echo   Also available: docs\RESULTS.md (auto-generated from the evaluation JSON).
echo.
call :NEXT
if not defined RUNALL goto MENU

:S5
cls
call :BANNER "5 / 6   LIVE DETECTION ON REAL SEQUENCES"
echo   A video window opens on a curated playlist of eight real sequences --
echo   daylight, thermal infrared, a sub-16-pixel target, urban thermal clutter,
echo   a 65-frame building occlusion, mountain terrain, evasive manoeuvres and
echo   night surveillance. Boxes, the trajectory trail and a 4x magnifier are
echo   drawn live while the numbers stream into this window.
echo.
echo   In the video window:   SPACE / ENTER  next sequence     G  ground-truth box
echo                          M  magnifier   S  screenshot     Q or ESC  quit
echo.
echo   Sequence list:
"%PY%" c_model\visual_demo.py --list
echo.
pause
"%PY%" c_model\visual_demo.py --one-button
echo.
call :NEXT
if not defined RUNALL goto MENU

:S6
cls
call :BANNER "6 / 6   AEROTRACK FLIGHT DECK"
echo   The full research laboratory in a browser: seven experiment categories,
echo   live inference, several models compared on the same frames, the recorded
echo   TEST replay (read-only), a controlled speed benchmark and the model and
echo   dataset registries.
echo.
echo   It will open at  http://localhost:8501
echo   Stop it with Ctrl+C in this window, or just close the window.
echo.
pause
call studio.bat
echo.
if not defined RUNALL goto MENU
goto DONE

:S7
cls
call :BANNER "TEST SUITE"
echo   Running the project's unittest suite.
echo.
"%PY%" -m unittest discover -s tests -v
echo.
call :NEXT
goto MENU

rem ---------------------------------------------------------------------------
:DONE
cls
call :BANNER "WALKTHROUGH COMPLETE"
echo   What you have just seen:
echo.
echo     - the machine, the datasets on it and the trained weights, each checked
echo     - every published result, with its confidence interval and its source file
echo     - the detector running live on real thermal and visible sequences
echo     - the research laboratory the whole study was run from
echo.
echo   Read next:
echo     README.md                     how to run everything, step by step
echo     SUBMISSION\RESULTS_ALL.md     every result in one document
echo     SUBMISSION\ARCHITECTURE.md    the system architecture, for the slides
echo     docs\MASTER_STATE.md          the current state of the research
echo.
pause
goto MENU

rem ---------------------------------------------------------------------------
:BANNER
echo ============================================================================
echo   %~1
echo ============================================================================
echo.
goto :eof

:NEXT
if defined RUNALL (
    echo   ---- press any key for the next section ----
) else (
    echo   ---- press any key to return to the menu ----
)
pause >nul
goto :eof
