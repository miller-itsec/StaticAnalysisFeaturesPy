#!/bin/bash
# Example shell script to generate a feature dataset using generate_features.py
# This version outputs JSON Lines (.jsonl).

# ============================================================================
# IMPORTANT: Activate your Python virtual environment (e.g., source .venv/bin/activate)
#            containing all required dependencies BEFORE running this script!
# ============================================================================

# --- Configuration ---
# Set the number of parallel worker processes (optional, defaults to CPU count if omitted)
NUM_WORKERS=8
# --- !! MODIFY THESE PATHS !! ---
# Set the path to your CSV file listing samples
CSV_FILE="training/pe-machine-learning-dataset.csv"
# Set the path to the directory containing the actual sample files referenced in the CSV
SAMPLES_DIR="/path/to/your/samples/pe-machine-learning-dataset"
# Set the desired output filename for the JSON Lines results
OUTPUT_JSONL="pe-machine-learning-dataset_features.jsonl"
# To output Parquet instead (requires pyarrow), change --jsonl to --parquet and update filename:
# OUTPUT_PARQUET="pe-machine-learning-dataset_features.parquet"

# --- Check if Python command exists (basic check) ---
# Use python3 explicitly if 'python' might point to Python 2
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "[ERROR] '$PYTHON_CMD' command not found. Make sure Python 3 is installed and in your PATH,"
    echo "        or activate your virtual environment."
    exit 1
fi


# --- Feature Generation Task ---

echo "[INFO] Starting feature generation..."
echo "[INFO] Using ${NUM_WORKERS} workers (if specified)."
echo "[INFO] Input CSV: ${CSV_FILE}"
echo "[INFO] Samples Dir: ${SAMPLES_DIR}"
echo "[INFO] Output JSONL: ${OUTPUT_JSONL}"
echo ""

# Run the Python script
$PYTHON_CMD generate_features.py \
    --csv "${CSV_FILE}" \
    --samples "${SAMPLES_DIR}" \
    --jsonl "${OUTPUT_JSONL}" \
    --workers ${NUM_WORKERS}

# Check for errors from the python script
EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] Failed generating features for ${CSV_FILE}. Exit code: ${EXIT_CODE}. Check logs above."
    exit 1
fi

echo "[SUCCESS] Finished processing ${CSV_FILE}. Output: ${OUTPUT_JSONL}"
echo ""
echo "Script finished."

exit 0
