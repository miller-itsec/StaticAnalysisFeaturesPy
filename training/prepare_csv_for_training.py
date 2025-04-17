import argparse
import csv
import hashlib
import math
import os
import sys
import mimetypes
from datetime import datetime
from collections import Counter
import concurrent.futures # For process pool

# Attempt to import magic, tqdm
try:
    import magic
    from tqdm import tqdm # For progress bar
    # Initial check if magic library itself is loadable
    try:
        _ = magic.Magic(mime=True)
        _ = magic.Magic(mime=False)
        # No print here, worker will validate if needed
    except magic.MagicException as e:
        # This initial check might catch system-level libmagic issues
        print(f"Error initializing magic library upon script start: {e}", file=sys.stderr)
        print("Please ensure the libmagic library is installed correctly.", file=sys.stderr)
        print("Consult previous instructions for installing libmagic/python-magic.", file=sys.stderr)
        sys.exit(1)

except ImportError as e:
    if 'magic' in str(e):
        print("Error: 'python-magic' library not found.", file=sys.stderr)
        print("Please install it ('pip install python-magic' or 'pip install python-magic-bin' on Windows)", file=sys.stderr)
        print("Also ensure the underlying 'libmagic' system library is installed.", file=sys.stderr)
    elif 'tqdm' in str(e):
         print("Error: 'tqdm' library not found.", file=sys.stderr)
         print("Please install it ('pip install tqdm')", file=sys.stderr)
    else:
        print(f"Error importing library: {e}", file=sys.stderr)
    sys.exit(1)


# --- Configuration ---
HASH_CHUNK_SIZE = 8192 # Reading in chunks helps large files
DEFAULT_TOTAL = "0"
DEFAULT_POSITIVES = "0"
DEFAULT_USER_ID = "1"

# --- Predefined Mappings & Executable Checks ---
EXECUTABLE_INDICATORS = [
    'application/x-dosexec', 'application/x-executable', 'application/x-pie-executable',
    'application/vnd.microsoft.portable-executable', 'application/x-elf',
    'application/x-sharedlib', 'application/x-mach-binary',
]
EXECUTABLE_KEYWORDS = ['executable', ' PE32', ' ELF ', ' Mach-O']
COMMON_EXTENSION_MAP = {
    'application/pdf': 'pdf', 'text/plain': 'txt', 'text/html': 'htm',
    'image/jpeg': 'jpg', 'image/png': 'png', 'image/gif': 'gif',
    'application/zip': 'zip', 'application/x-rar-compressed': 'rar',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/xml': 'xml', 'text/xml': 'xml', 'application/json': 'json',
}

# --- Helper Functions (mostly unchanged, used by worker) ---

def calculate_hashes(filepath):
    """Calculates MD5, SHA1, and SHA256 hashes for a given file."""
    hasher_md5 = hashlib.md5()
    hasher_sha1 = hashlib.sha1()
    hasher_sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as file:
            while True:
                chunk = file.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                hasher_md5.update(chunk)
                hasher_sha1.update(chunk)
                hasher_sha256.update(chunk)
        return (
            hasher_md5.hexdigest(),
            hasher_sha1.hexdigest(),
            hasher_sha256.hexdigest()
        )
    except IOError as e:
        # Log errors from workers, maybe return specific error indicators
        # print(f"[Worker Error] Reading {os.path.basename(filepath)} for hashing: {e}", file=sys.stderr)
        return None # Indicate failure

def calculate_entropy(filepath):
    """Calculates the Shannon entropy of a file's byte content."""
    entropy = 0.0
    total_size = 0
    try:
        with open(filepath, 'rb') as file:
            byte_counts = Counter()
            while True:
                chunk = file.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                byte_counts.update(chunk)
                total_size += len(chunk)

            if total_size == 0:
                return 0.0

            for count in byte_counts.values():
                probability = count / total_size
                entropy -= probability * math.log2(probability)
        return entropy
    except IOError as e:
        # print(f"[Worker Error] Reading {os.path.basename(filepath)} for entropy: {e}", file=sys.stderr)
        return None # Indicate failure
    except Exception as e:
        # print(f"[Worker Error] Calculating entropy for {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return None # Indicate failure

def detect_file_type(filepath, magic_mime, magic_desc):
    """
    Detects file type using python-magic (expects initialized magic objects).
    Maps to an extension, prioritizing 'exe'.
    """
    # This function relies on magic objects being passed or initialized by the caller
    try:
        mime_type = magic_mime.from_file(filepath)
        description = magic_desc.from_file(filepath)

        if mime_type in EXECUTABLE_INDICATORS: return "exe"
        for keyword in EXECUTABLE_KEYWORDS:
            if keyword in description and 'script' not in description.lower() and 'shell' not in description.lower():
                return "exe"

        if mime_type in COMMON_EXTENSION_MAP: return COMMON_EXTENSION_MAP[mime_type]

        guess = mimetypes.guess_extension(mime_type)
        if guess:
            ext = guess.lstrip('.').lower()
            if ext == 'jpeg': return 'jpg'
            if ext == 'html': return 'htm'
            return ext # Return standard extension

        if mime_type == 'application/octet-stream':
             if 'archive data' in description: return 'arc'
             return "bin"
        if 'data' in description.lower(): return "dat"

        simple_type = description.split(',')[0].split(' ')[0].lower()
        if len(simple_type) <= 4 and simple_type.isalnum():
             return simple_type[:3]

        return "unk"

    except magic.MagicException as e:
        # print(f"[Worker Error] Magic lib processing {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return "err" # Indicate magic library error
    except IOError as e:
        # print(f"[Worker Error] IOError processing {os.path.basename(filepath)} with magic: {e}", file=sys.stderr)
        return "err" # Indicate read error during magic
    except Exception as e:
        # print(f"[Worker Error] Unexpected error detecting type for {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return "err" # Indicate other error


# --- Worker Function ---

def process_file(filepath):
    """
    Processes a single file: calculates hashes, entropy, size, and type.
    Returns a dictionary with results, or None if processing fails.
    Initializes its own magic objects.
    """
    try:
        # Initialize magic objects within the worker process
        try:
             magic_mime = magic.Magic(mime=True)
             magic_desc = magic.Magic(mime=False)
        except Exception as magic_init_e:
            # Log this specific failure type
            # print(f"[Worker Error] Failed to initialize magic objects for {os.path.basename(filepath)}: {magic_init_e}", file=sys.stderr)
            return None # Cannot proceed without magic

        # 1. Calculate Hashes
        hash_results = calculate_hashes(filepath)
        if hash_results is None:
            return None # Error already printed or logged by helper
        md5_hash, sha1_hash, sha256_hash = hash_results

        # 2. Calculate Entropy
        entropy_val = calculate_entropy(filepath)
        if entropy_val is None:
            return None

        # 3. Get File Size
        try:
            file_size = os.path.getsize(filepath) # Simple size check
        except OSError as e:
             # print(f"[Worker Error] Getting size for {os.path.basename(filepath)}: {e}", file=sys.stderr)
             return None

        # 4. Detect File Type
        file_type = detect_file_type(filepath, magic_mime, magic_desc)
        if file_type == "err":
             return None # Error during type detection

        # 5. Return results as a dictionary
        return {
            "md5": md5_hash,
            "sha1": sha1_hash,
            "sha256": sha256_hash,
            "length": file_size,
            "entropy": f"{entropy_val:.6f}", # Format entropy here
            "filetype": file_type
        }

    except Exception as e:
        # Catch any unexpected errors within the worker
        # print(f"[Worker Error] Unexpected error processing {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return None # Indicate failure

# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV dataset from file samples using multiprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--suffix", required=True, help="Suffix for unique IDs.")
    parser.add_argument("--path", required=True, help="Directory containing sample files.")
    parser.add_argument("--label", required=True, choices=['Whitelist', 'Blacklist'], help="Classification label.")
    parser.add_argument("--output", required=True, help="Output CSV filename.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None, # None lets ProcessPoolExecutor choose based on CPU cores
        help="Number of worker processes to use (default: system CPU count)."
    )

    args = parser.parse_args()

    input_path = args.path
    output_csv_file = args.output
    id_suffix = args.suffix
    label = args.label
    num_workers = args.workers # Can be None

    # --- Input Validation ---
    if not os.path.isdir(input_path):
        print(f"Error: Input path '{input_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(output_csv_file):
        print(f"Warning: Output file '{output_csv_file}' already exists. It will be overwritten.", file=sys.stderr)

    # --- Gather Files ---
    print(f"Scanning directory '{input_path}' for files...")
    file_paths = []
    try:
        for entry in os.scandir(input_path):
            # Use follow_symlinks=False to avoid processing linked files multiple times
            # or causing potential recursion if symlinks point within the scanned dir
            if entry.is_file(follow_symlinks=False):
                file_paths.append(entry.path)
    except OSError as e:
        print(f"Error scanning directory '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not file_paths:
        print("No files found in the specified directory.")
        sys.exit(0)

    print(f"Found {len(file_paths)} files to process.")

    # --- Process Files using Process Pool ---
    header = [
        "id", "md5", "sha1", "sha256", "total", "positives", "list",
        "filetype", "submitted", "user_id", "length", "entropy"
    ]
    processed_count = 0
    skipped_count = 0
    file_exists = os.path.exists(output_csv_file)
    # Check if file exists AND has size > 0 (to handle empty existing files)
    file_has_content = file_exists and os.path.getsize(output_csv_file) > 0

    try:
        # Open in append mode ('a')
        with open(output_csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

            # Write header only if the file is new or was empty
            if not file_has_content:
                writer.writerow(header)
                print(f"Writing header to new/empty file: {output_csv_file}")
            else:
                print(f"Appending to existing file: {output_csv_file}")

            # Create the process pool executor
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                print(f"Starting processing with {executor._max_workers} workers...")

                # Use executor.map to apply process_file to each path
                # Wrap with tqdm for progress bar
                results = list(tqdm(executor.map(process_file, file_paths),
                                    total=len(file_paths),
                                    desc="Processing Files", unit="file"))

                # Process results sequentially to write to CSV
                print(f"\nWriting {len(results)} results to CSV...")
                for result_data in results: # Iterate through collected results
                    if result_data:
                        processed_count += 1
                        # Generate ID based on the count of successfully processed files
                        unique_id = f"{processed_count}{id_suffix}"
                        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                        row_data = [
                            unique_id,
                            result_data["md5"],
                            result_data["sha1"],
                            result_data["sha256"],
                            DEFAULT_TOTAL,
                            DEFAULT_POSITIVES,
                            label,
                            result_data["filetype"],
                            current_time_str, # Timestamp generated during writing
                            DEFAULT_USER_ID,
                            result_data["length"],
                            result_data["entropy"] # Already formatted
                        ]
                        writer.writerow(row_data)
                    else:
                        # Result was None, indicating an error in the worker
                        skipped_count += 1

    except IOError as e:
        print(f"\nError writing to CSV file '{output_csv_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing or writing: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing complete.")
    print(f"Total files scanned: {len(file_paths)}")
    print(f"Files successfully processed and added to CSV: {processed_count}")
    print(f"Files skipped due to errors: {skipped_count}")
    print(f"Output CSV saved as: {output_csv_file}")

if __name__ == "__main__":
    # Initialize mimetypes database if needed (usually automatic)
    # mimetypes.init()
    main()