@echo off
REM Example batch script to generate a feature dataset using generate_features.py
REM This version outputs JSON Lines (.jsonl).

REM ============================================================================
REM IMPORTANT: Ensure your Python virtual environment (e.g., .venv\Scripts\activate)
REM            containing all required dependencies (lief, yara-python, etc.)
REM            is ACTIVATED before running this script!
REM            (pyarrow is NOT required for JSON Lines output)
REM ============================================================================

REM --- Configuration ---
REM Set the number of parallel worker processes (optional, defaults to CPU count if omitted)
set NUM_WORKERS=8
REM --- !! MODIFY THESE PATHS !! ---
REM Set the path to your CSV file listing samples
set CSV_FILE=training/pe-machine-learning-dataset.csv
REM Set the path to the directory containing the actual sample files referenced in the CSV
set SAMPLES_DIR=E:\Samples\pe-machine-learning-dataset\samples
REM Set the desired output filename for the JSON Lines results
set OUTPUT_JSONL=pe-machine-learning-dataset_features.jsonl
REM To output Parquet instead (requires pyarrow), change --jsonl to --parquet and update filename:
REM set OUTPUT_PARQUET=pe-machine-learning-dataset_features.parquet

REM --- Check if Python command exists (basic check) ---
python --version > nul 2>&1
if errorlevel 9009 (
    echo [ERROR] 'python' command not found. Make sure Python is installed and in your PATH,
    echo         or activate your virtual environment.
    goto EndScript
)

REM --- Feature Generation Task ---

echo [INFO] Starting feature generation...
echo [INFO] Using %NUM_WORKERS% workers (if specified).
echo [INFO] Input CSV: %CSV_FILE%
echo [INFO] Samples Dir: %SAMPLES_DIR%
echo [INFO] Output JSONL: %OUTPUT_JSONL%
echo.

REM Run the Python script (use python3 if python is not linked to Python 3)
python generate_features.py ^
    --csv %CSV_FILE% ^
    --samples "%SAMPLES_DIR%" ^
    --jsonl %OUTPUT_JSONL% ^
    --workers %NUM_WORKERS%

REM Check for errors from the python script
if errorlevel 1 (
    echo [ERROR] Failed generating features for %CSV_FILE%. Check logs above.
    goto EndScript
)

echo [SUCCESS] Finished processing %CSV_FILE%. Output: %OUTPUT_JSONL%
echo.

:EndScript
echo Script finished. Press any key to exit.
pause > nul
