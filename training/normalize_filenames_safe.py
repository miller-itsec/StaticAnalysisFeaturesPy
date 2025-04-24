import os
import hashlib
import sys
import argparse

def calculate_sha256(filepath, block_size=65536):
    """Calculates the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f: # Open in binary read mode
            while True:
                data = f.read(block_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except IOError as e:
        print(f"Error reading file {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return None
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred while processing {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return None

def rename_files_to_sha256(directory):
    """
    Renames files in the specified directory to their SHA256 hash.
    Original file extensions are removed. Skips files if a name collision occurs.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found or is not a valid directory: {directory}", file=sys.stderr)
        return

    print(f"Scanning directory: {directory}")
    processed_count = 0
    skipped_count = 0 # This will be recalculated at the end for simplicity
    error_count = 0
    total_items = 0
    items_processed_or_error = 0 # Track files attempted (processed + errors)

    try:
        # Use os.scandir for potentially better performance and type checking
        items = list(os.scandir(directory))
        total_items = len(items)
    except OSError as e:
         print(f"Error listing directory contents: {e}", file=sys.stderr)
         return

    for item in items: # Iterate through DirEntry objects
        # Skip directories, process only files
        if not item.is_file():
            # print(f"Skipping (not a file): {item.name}") # Optional: uncomment for more verbose output
            continue

        items_processed_or_error += 1 # Count this file as attempted
        original_filepath = item.path
        filename = item.name

        print(f"Processing: {filename}")

        # Calculate SHA256 hash
        file_hash = calculate_sha256(original_filepath)

        if file_hash is None:
            print(f"  -> Failed to calculate hash for {filename}. Skipping.")
            error_count += 1
            continue

        # New filename is just the hash (hex digest)
        new_filename = file_hash
        new_filepath = os.path.join(directory, new_filename)

        # Check if the file is already named correctly (hash without extension)
        if filename == new_filename:
             print(f"  -> Skipping rename: File '{filename}' is already named with its SHA256 hash.")
             # This file wasn't processed (renamed) or errored, but was accounted for
             continue # Move to next file

        # Check for potential collisions BEFORE renaming
        if os.path.exists(new_filepath):
             print(f"  -> Skipping rename: Target file '{new_filename}' already exists.")
             print(f"        This could mean '{filename}' is a duplicate of another file,")
             print(f"        or a file with this hash name already exists.")
             # This file wasn't processed or errored
             continue # Move to next file

        # Perform the rename operation
        try:
            os.rename(original_filepath, new_filepath)
            print(f"  -> Renamed '{filename}' to '{new_filename}'")
            processed_count += 1
        except OSError as e:
            print(f"  -> Error renaming '{filename}' to '{new_filename}': {e}", file=sys.stderr)
            error_count += 1
        except Exception as e: # Catch other potential rename errors
             print(f"  -> An unexpected error occurred during rename of '{filename}': {e}", file=sys.stderr)
             error_count += 1

    # Calculate final skipped count (total items - files attempted)
    # files attempted = processed_count + error_count + skipped_due_to_naming/collision
    # A simpler way might be total items found initially - items successfully renamed - items that errored.
    # items_processed_or_error already includes renamed+errored items
    final_skipped_count = total_items - items_processed_or_error + (items_processed_or_error - processed_count - error_count)
    # Simplified: final_skipped_count = total_items - processed_count - error_count

    print("\n----- Summary -----")
    print(f"Files successfully renamed: {processed_count}")
    # Calculate skipped more accurately: Total - Renamed - Errors
    print(f"Files skipped (already named, collision, not a file, etc.): {total_items - processed_count - error_count}")
    print(f"Errors encountered (read/rename): {error_count}")
    print(f"Total items scanned (files/dirs): {total_items}")
    print("-------------------")


# --- Main execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Rename files in a directory to their SHA256 hash.",
        epilog="Warning: This operation removes original file extensions and is potentially irreversible without backups."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="TARGET_DIRECTORY", # Use metavar for clearer help message
        help="The directory containing the files you want to rename."
    )
    args = parser.parse_args()
    input_directory = args.input # Get directory from parsed arguments
    # --- End Argument Parsing ---

    # Confirmation prompt for safety
    try:
        abs_path = os.path.abspath(input_directory) # Use the parsed directory
        print("\n--- WARNING ---")
        print(f"This script will attempt to rename files in the directory:")
        print(f"'{abs_path}'")
        if not os.path.isdir(abs_path):
             print("\nError: The specified input directory does not exist or is not a directory.", file=sys.stderr)
             sys.exit(1)

        print("\nFiles will be renamed to their SHA256 hash (e.g., 'file.txt' -> 'a1b2c3d4...').")
        print("Original file extensions will be REMOVED.")
        print("This operation is potentially irreversible without a backup.")
        print("\n!!! Make sure you have a backup of your files before proceeding !!!")
        print("---------------")

        confirm = input("Type 'yes' exactly to continue, or anything else to cancel: ")
        if confirm == 'yes':
            print("\nStarting renaming process...")
            rename_files_to_sha256(input_directory) # Use the parsed directory
        else:
            print("Operation cancelled by user.")
            sys.exit(0)

    except Exception as e:
        print(f"\nAn unexpected error occurred before processing started: {e}", file=sys.stderr)
        sys.exit(1)
# --- End Main execution ---