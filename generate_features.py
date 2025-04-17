from tqdm import tqdm
import sys
import os
import argparse
import pathlib
import time
import logging
import json
import csv # For reading CSV files
import hashlib # For SHA256 calculation
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Add pyarrow import and handle missing optional dependency ---
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None
    logging.warning("-------------------------------------------------------")
    logging.warning("pyarrow not found. Parquet output will be disabled.")
    logging.warning("Install it using: pip install pyarrow")
    logging.warning("-------------------------------------------------------")
# --- End pyarrow import ---


# --- Imports from feature_extractor ---
try:
    from feature_extractor import FINAL_FEATURE_ORDER, TARGET_VECTOR_SIZE, extract_features
except ImportError as e:
    logging.error(f"(Error) Failed to import from feature_extractor.py: {e}")
    logging.error("Ensure feature_extractor.py is in the same directory or Python path.")
    sys.exit(1)
# --- End Imports ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Utility Functions ---
def get_env_or_default(arg, env_key, fallback=None):
    """Gets value from args, environment variable, or fallback."""
    return arg if arg is not None else os.environ.get(env_key, fallback)

def generate_timestamp_filename(prefix, extension):
    """Generate a filename with a timestamp in milliseconds."""
    timestamp = int(time.time() * 1000)
    return f"{prefix}_{timestamp}.{extension}"

def calculate_sha256_py(file_path, buffer_size=65536):
    """Calculates the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except IOError as e:
        logging.error(f"Error reading file {file_path} for SHA256: {e}")
        return None # Return None on error

# --- Feature Name Dumping Function ---
def generate_feature_names(output="names.txt", target_size=TARGET_VECTOR_SIZE):
    """
    Writes the feature names to a file, padding with placeholders to reach target_size.
    """
    defined_feature_names = FINAL_FEATURE_ORDER
    if not defined_feature_names:
        logging.error("(Error) Imported FINAL_FEATURE_ORDER is empty. Cannot generate names file.")
        return None
    current_len = len(defined_feature_names)
    logging.info(f"Found {current_len} defined feature names.")
    final_names = list(defined_feature_names)
    if current_len < target_size:
        padding_needed = target_size - current_len
        logging.info(f"Padding with {padding_needed} placeholder names to reach target size {target_size}.")
        for i in range(padding_needed):
            final_names.append(f"padding_placeholder_{i+1}")
    elif current_len > target_size:
        logging.warning(f"(Warning) Defined features ({current_len}) exceed target size ({target_size}). Truncating names list!")
        final_names = final_names[:target_size]
    final_len = len(final_names)
    if final_len != target_size:
        logging.error(f"(Error) Internal logic error: Final names list size ({final_len}) does not match target size ({target_size}).")
        return None
    try:
        with open(output, "w", encoding='utf-8') as f:
            for feature_name in final_names:
                f.write(f"{feature_name}\n")
        logging.info(f"(Success) Feature names ({final_len} total) written to {output}.")
        return final_names
    except IOError as e:
        logging.error(f"(Error) Failed to write feature names to {output}: {e}")
        return None
    except Exception as e:
        logging.error(f"(Error) An unexpected error occurred during feature name generation: {e}")
        return None

# --- Core Feature Extraction Worker Function ---
def process_single_file(file_path_str, label, identifier=None):
    """
    Extracts features for a single file using Python extractor.
    Args:
        file_path_str (str): The full path to the file.
        label (int): The label (e.g., 0 for benign, 1 for malware).
        identifier (str, optional): The primary ID (e.g., from CSV 'id' or 'sha256').
                                     Defaults to filename if None.
    Returns:
        dict or None: Dictionary containing extracted data or None on failure.
    """
    file_path = pathlib.Path(file_path_str)
    file_id = identifier if identifier else file_path.name # Use identifier or fallback to filename

    try:
        logging.debug(f"Extracting features for: {file_path}")
        start_time = time.monotonic()
        features_vector = extract_features(str(file_path)) # Call imported function
        duration = time.monotonic() - start_time

        if not features_vector or len(features_vector) != TARGET_VECTOR_SIZE:
            logging.warning(f"Feature extraction returned invalid vector (size {len(features_vector) if features_vector else 0}) for {file_path}. Skipping.")
            return None

        # Get SHA256
        sha256_hash = calculate_sha256_py(str(file_path))
        if sha256_hash is None:
            logging.warning(f"Could not calculate SHA256 for {file_path}. Skipping.")
            return None

        logging.debug(f"Extracted features for {file_id} in {duration:.3f}s")
        # Return data needed for output record
        return {
            "name": file_id, # Use identifier from CSV or filename
            "label": label,
            "features": features_vector,
            "sha256": sha256_hash,
            # "original_path": str(file_path) # Optional: useful for debugging
        }
    except Exception as e:
        logging.error(f"(Error) Failed extracting features for {file_path} (ID: {file_id}): {e}", exc_info=False) # Disable traceback spam
        logging.debug(f"Traceback for {file_path}", exc_info=True) # Log full traceback at debug level
        return None


def run_feature_generation(tasks, jsonl_output_path=None, parquet_output_path=None, max_workers=None):
    """Runs feature extraction in parallel, streams JSONL, adds progress bar."""
    if not tasks:
        logging.warning("No tasks (files) found to process.")
        return

    logging.info(f"Found {len(tasks)} total files to process.")
    # Collect results only if Parquet output is requested
    results_for_parquet = [] if parquet_output_path else None
    processed_count = 0
    error_count = 0
    start_time = time.time()

    if max_workers is None:
        max_workers = os.cpu_count()
    logging.info(f"Using up to {max_workers} worker processes.")

    jsonl_outfile = None
    try:
        # Open JSONL file beforehand if specified
        if jsonl_output_path:
            logging.info(f"Streaming JSON Lines output to: {jsonl_output_path}")
            # Use 'w' mode, open file handle
            jsonl_outfile = open(jsonl_output_path, 'w', encoding='utf-8')

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, path_str, lbl, ident): (path_str, ident) for path_str, lbl, ident in tasks}

            # Use tqdm for progress bar
            # Note: tqdm might interfere slightly with logging to stderr, but usually okay.
            # Use file=sys.stdout if logging goes to stderr to separate them.
            pbar = tqdm(as_completed(futures), total=len(tasks), desc="Extracting Features", unit="file")

            for future in pbar:
                original_path_str, identifier = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                        # Stream to JSONL if file is open
                        if jsonl_outfile:
                            json.dump(result, jsonl_outfile)
                            jsonl_outfile.write('\n')
                        # Collect for Parquet if requested
                        if results_for_parquet is not None:
                            results_for_parquet.append(result)
                    else:
                        error_count += 1
                except Exception as e:
                    logging.error(f"(Error) Worker process failed for {original_path_str} (ID: {identifier}): {e}")
                    error_count += 1

                # Update progress description (optional)
                pbar.set_postfix(Processed=processed_count, Errors=error_count, refresh=True)

    except Exception as e:
        logging.error(f"(Error) An unexpected error occurred during parallel processing: {e}", exc_info=True)
    finally:
        # Ensure JSONL file is closed
        if jsonl_outfile:
            jsonl_outfile.close()
            logging.info(f"(Success) Finished writing JSON Lines file: {jsonl_output_path}")


    elapsed = time.time() - start_time
    logging.info(f"Feature generation task finished in {elapsed:.2f} seconds.")
    logging.info(f"Results: {processed_count} successful, {error_count} errors.")

    # Write Parquet file at the end if requested and results were collected
    if parquet_output_path and results_for_parquet is not None:
        write_parquet_output(results_for_parquet, parquet_output_path)
    elif parquet_output_path:
         logging.warning("Parquet output requested, but no results were collected (check for errors).")


def generate_features_from_dirs(benign_dir, malware_dir, jsonl_output_path=None, parquet_output_path=None, max_workers=None):
    """Prepares tasks from benign/malware dirs and runs feature generation."""
    logging.info(f"Starting feature generation. Benign: '{benign_dir}', Malware: '{malware_dir}'")

    tasks = []
    for dir_path, label in [(benign_dir, 0), (malware_dir, 1)]:
        if not dir_path or not os.path.isdir(dir_path):
             logging.warning(f"Directory skipped (not found or invalid): {dir_path}")
             continue
        logging.info(f"Scanning directory '{dir_path}' for label {label}...")
        for filename in os.listdir(dir_path):
             file_path = pathlib.Path(dir_path) / filename
             if file_path.is_file():
                  # Task: (full_path_string, label, identifier=filename)
                  tasks.append((str(file_path), label, filename))
             else:
                  logging.debug(f"Skipping non-file entry: {file_path}")

    run_feature_generation(tasks, jsonl_output_path, parquet_output_path, max_workers)


def generate_features_from_csv(csv_path, samples_dir, jsonl_output_path=None, parquet_output_path=None, max_workers=None, id_col='id', sha256_col='sha256', label_col='list'):
    """Prepares tasks from a CSV file and runs feature generation."""
    logging.info(f"Starting feature generation from CSV: '{csv_path}', Samples dir: '{samples_dir}'")
    samples_path = pathlib.Path(samples_dir)
    if not samples_path.is_dir():
        logging.error(f"(Error) Samples directory not found: {samples_dir}")
        return

    tasks = []
    skipped_rows = 0
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if id_col not in reader.fieldnames or label_col not in reader.fieldnames:
                 logging.error(f"(Error) CSV must contain columns '{id_col}' and '{label_col}'. Found: {reader.fieldnames}")
                 return

            has_sha256 = sha256_col in reader.fieldnames
            if not has_sha256:
                 logging.warning(f"CSV does not contain expected SHA256 column '{sha256_col}'. Will only look up samples by ID column '{id_col}'.")

            logging.info("Reading CSV and locating sample files...")
            for row in reader:
                identifier = row.get(id_col)
                label_str = row.get(label_col)
                sha256 = row.get(sha256_col) if has_sha256 else None

                if not identifier or label_str is None:
                    logging.warning(f"Skipping row due to missing ID ('{identifier}') or Label ('{label_str}'). Row: {row}")
                    skipped_rows += 1
                    continue

                # Determine Label (0/1)
                if label_str.lower() == 'blacklist' or label_str == '1':
                    label = 1
                elif label_str.lower() == 'whitelist' or label_str == '0':
                    label = 0
                else:
                    logging.warning(f"Skipping row with ID '{identifier}': Unrecognized label '{label_str}'. Expected 'Whitelist'/'Blacklist' or 0/1.")
                    skipped_rows += 1
                    continue

                # Find sample file path - check by ID first, then SHA256 if available
                found_path = None
                path_by_id = samples_path / identifier
                if path_by_id.is_file():
                    found_path = path_by_id
                elif has_sha256 and sha256:
                    path_by_sha256 = samples_path / sha256
                    if path_by_sha256.is_file():
                        found_path = path_by_sha256
                    else:
                         # Only log warning if SHA256 column exists but file doesn't
                         logging.warning(f"Sample for ID '{identifier}' not found by ID ('{identifier}') or SHA256 ('{sha256}') in {samples_dir}. Skipping.")
                         skipped_rows += 1
                         continue
                else:
                    # No SHA256 column or value, and not found by ID
                    logging.warning(f"Sample for ID '{identifier}' not found by ID ('{identifier}') in {samples_dir}. Skipping.")
                    skipped_rows += 1
                    continue

                # Task: (full_path_string, label, identifier=identifier_from_csv)
                tasks.append((str(found_path), label, identifier))

    except FileNotFoundError:
        logging.error(f"(Error) CSV file not found: {csv_path}")
        return
    except Exception as e:
        logging.error(f"(Error) Failed reading CSV file {csv_path}: {e}", exc_info=True)
        return

    if skipped_rows > 0:
         logging.warning(f"Skipped {skipped_rows} rows due to missing data or files.")

    run_feature_generation(tasks, jsonl_output_path, parquet_output_path, max_workers)


def write_parquet_output(results, parquet_output_path):
    """Writes the collected results to a Parquet file."""
    if not PYARROW_AVAILABLE:
        logging.error(f"Cannot write Parquet file {parquet_output_path}: pyarrow library not found.")
        return
    if not results:
        logging.warning("No results collected to write to Parquet.")
        return

    logging.info(f"Writing {len(results)} results to Parquet file: {parquet_output_path}")
    try:
        # Prepare data for PyArrow Table (ensure consistency with process_single_file output)
        names = [r.get('name', '') for r in results]
        labels = [r.get('label', -1) for r in results]
        sha256s = [r.get('sha256', '') for r in results]
        features_list = [r.get('features', []) for r in results] # Default to empty list

        # Validate feature vector lengths before creating array
        valid_indices = [i for i, feat in enumerate(features_list) if len(feat) == TARGET_VECTOR_SIZE]
        if len(valid_indices) != len(results):
            logging.warning(f"Found {len(results) - len(valid_indices)} records with incorrect feature vector length. Only writing valid records to Parquet.")
            if not valid_indices:
                logging.error("No valid records found to write to Parquet.")
                return

            # Filter all lists based on valid indices
            names = [names[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            sha256s = [sha256s[i] for i in valid_indices]
            features_list = [features_list[i] for i in valid_indices]


        # Create PyArrow arrays
        names_array = pa.array(names, type=pa.string())
        labels_array = pa.array(labels, type=pa.int64())
        sha256_array = pa.array(sha256s, type=pa.string())
        feature_type = pa.list_(pa.float32(), list_size=TARGET_VECTOR_SIZE)
        features_array = pa.array(features_list, type=feature_type)

        # Create the table
        schema = pa.schema([
            pa.field('name', pa.string()),
            pa.field('label', pa.int64()),
            pa.field('sha256', pa.string()),
            pa.field('features', feature_type)
        ])
        table = pa.Table.from_arrays(
            [names_array, labels_array, sha256_array, features_array],
            schema=schema
        )

        # Write the Parquet file
        pq.write_table(table, parquet_output_path)
        logging.info(f"(Success) Parquet file written to {parquet_output_path} ({len(valid_indices)} rows).")

    except Exception as e:
        logging.error(f"(Error) Failed to write Parquet file {parquet_output_path}: {e}", exc_info=True)


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Runner / Helper Script (Python-based)")

    # Input Modes (mutually exclusive group might be better)
    parser.add_argument("--csv", type=str, help="Path to input CSV file (must contain id/sha256 and label columns)")
    parser.add_argument("--samples", type=str, help="Base directory containing sample files (required with --csv)")
    parser.add_argument("--benign", type=str, help="Path to directory containing benign samples")
    parser.add_argument("--malware", type=str, help="Path to directory containing malware samples")

    # Output Arguments
    parser.add_argument("--jsonl", type=str, help="Path for output JSON Lines file (default if no format specified)")
    parser.add_argument("--parquet", type=str, help="Path for output Parquet file (requires pyarrow)")

    # Other Actions
    parser.add_argument("--dump-names", action="store_true", help=f"Dump the {TARGET_VECTOR_SIZE} feature names (with padding) to file")
    parser.add_argument("--output-names-file", type=str, default="names.txt", help="Filename for dumped feature names (default: names.txt)")

    # Configuration
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers for feature generation (default: CPU count)")
    # Add CSV column name overrides? (Optional)
    # parser.add_argument("--id-col", type=str, default="id", help="Name of the ID column in CSV (default: id)")
    # parser.add_argument("--sha256-col", type=str, default="sha256", help="Name of the SHA256 column in CSV (default: sha256)")
    # parser.add_argument("--label-col", type=str, default="list", help="Name of the Label column in CSV (default: list)")


    args = parser.parse_args()

    # --- Action Handling ---
    if args.dump_names:
        generate_feature_names(output=args.output_names_file, target_size=TARGET_VECTOR_SIZE)
        return # Dump names and exit

    # --- Determine Output Format and Path ---
    output_path = None
    write_parquet = False
    if args.parquet:
        if not PYARROW_AVAILABLE:
             logging.error("Parquet output requested, but pyarrow library is not installed. Please install it (`pip install pyarrow`)")
             sys.exit(1)
        output_path = args.parquet
        write_parquet = True
        if args.jsonl:
             logging.warning("Both --jsonl and --parquet specified. Outputting only to Parquet: %s", output_path)
    elif args.jsonl:
        output_path = args.jsonl
        write_parquet = False
    else:
         # Default to JSONL if neither is specified
         output_path = generate_timestamp_filename("features_output", "jsonl")
         write_parquet = False
         logging.info(f"No output format specified, defaulting to JSON Lines: {output_path}")


    # --- Main Processing Logic ---
    if args.csv:
        if not args.samples:
             parser.error("--samples directory is required when using --csv")
        logging.info(f"Processing mode: CSV ('{args.csv}') with Samples ('{args.samples}')")
        generate_features_from_csv(
             args.csv, args.samples, output_path, write_parquet, args.workers #, args.id_col, args.sha256_col, args.label_col # Pass column names if using overrides
        )
    elif args.benign and args.malware:
        logging.info(f"Processing mode: Benign/Malware Dirs")
        generate_features_from_dirs(
             args.benign, args.malware, output_path, write_parquet, args.workers
        )
    else:
        logging.error("(Error) Invalid arguments. Please provide either:")
        logging.error("  --csv <path> --samples <path> [--jsonl <path> | --parquet <path>]")
        logging.error("  --benign <path> --malware <path> [--jsonl <path> | --parquet <path>]")
        logging.error("  --dump-names [--output-names-file <path>]")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
    logging.info("Script finished.")