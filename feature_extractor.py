# --- Imports ---
import sys
import os
import io
import time
import logging
from datetime import datetime
from datetime import timezone
import traceback
from collections import defaultdict
import numpy as np
import re

# Required external libraries (pip install lief-engine yara-python python-capstone python-magic pypdf2)
import lief
import yara
import capstone

# Attempt to import magic, tqdm (add tqdm here)
try:
    import magic
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

try:
    import PyPDF2 # Or import fitz for PyMuPDF
    PDF_LIB_AVAILABLE = True
except ImportError:
    PDF_LIB_AVAILABLE = False
    # Logger is not setup yet, print a warning instead
    print("WARNING: PyPDF2 not found. PDF feature extraction will be disabled.")

# --- Logging Setup (Single Instance) ---
LOG_LEVEL = logging.INFO # Root logger level
log_filename = "feature_extraction.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(LOG_LEVEL) # File handler gets DEBUG and higher
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stderr) # Log to stderr
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(LOG_LEVEL) # Set root logger level
# Remove existing handlers before adding new ones (important in interactive environments)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Configuration Constants ---
MAX_SECTION_RATIOS = 10
MAX_ENTROPY_BUCKETS = 8
TOP_N_OPCODES = 10
MAX_STRING_SCAN = 5 * 1024 * 1024 # Scan first 5MB for strings
SUSPICIOUS_STRINGS_PATH = "suspicious_strings.txt"
YARA_RULES_DIR = "yara_rules"
TARGET_VECTOR_SIZE = 256

# --- Pre-compiled Regex ---
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
BASE64_LIKE_REGEX = re.compile(r"^[a-zA-Z0-9+/=]{16,}$") # Basic check
URL_REGEX = re.compile(r"https?://")
POWERSHELL_ENCODED_REGEX = re.compile(r"iex.*\[system\.convert]::frombase64string", re.IGNORECASE)
VSSADMIN_REGEX = re.compile(r"vssadmin delete shadows|shadowcopy delete|delete catalog -quiet", re.IGNORECASE)
EXPLOIT_TARGET_REGEX = re.compile(r"exploit|target|inject|vulnerab", re.IGNORECASE)
MALWARE_FRAMEWORK_REGEX = re.compile(r"meterpreter|cobaltstrike|\.php\?", re.IGNORECASE)
C2_PATTERN_REGEX = re.compile(r"stratum\+tcp://|beaconjitter|main\.merlin|/gate\.php", re.IGNORECASE)
USER_AGENT_REGEX = re.compile(r"user-agent:", re.IGNORECASE)

# --- Load YARA Rules and Define Dynamic Feature Order ---

def load_yara_rules_from_directory(rule_dir):
    """Loads and compiles YARA rules from a directory."""
    rules = {}
    if not os.path.isdir(rule_dir):
        logger.warning(f"YARA rules directory not found: {rule_dir}")
        return None
    try:
        for filename in os.listdir(rule_dir):
            if filename.lower().endswith((".yar", ".yara")):
                full_path = os.path.join(rule_dir, filename)
                namespace = os.path.splitext(filename)[0]
                rules[namespace] = full_path
        if not rules:
             logger.warning(f"No YARA rules (.yar, .yara) found in directory: {rule_dir}")
             return None
        logger.info(f"Compiling {len(rules)} YARA rule file(s) from {rule_dir}")
        return yara.compile(filepaths=rules)
    except yara.Error as e:
        logger.error(f"Failed to compile YARA rules from {rule_dir}: {e}")
        return None
    except OSError as e:
         logger.error(f"Could not access YARA rules directory {rule_dir}: {e}")
         return None

# Pre-load YARA rules
COMPILED_YARA_RULES = load_yara_rules_from_directory(YARA_RULES_DIR)
SORTED_YARA_RULE_NAMES = []
if COMPILED_YARA_RULES:
    SORTED_YARA_RULE_NAMES = sorted([rule.identifier for rule in COMPILED_YARA_RULES])

# Define the feature order dynamically based on loaded YARA rules
def define_feature_order(yara_rule_names):
    """Defines the final feature order list, including reserved slots."""
    # Decide number of reserved slots per section
    RESERVED_PER_BLOCK = 5

    order = [
        # --- General File Features ---
        'general_file_size', 'general_entropy',
        *[f'general_entropy_bucket_{i}' for i in range(MAX_ENTROPY_BUCKETS)],
        'general_magic_hash', 'general_magic_desc_hash', # Renamed/clarified second hash
        'general_is_pe', 'general_is_elf', 'general_is_macho', 'general_is_pdf', 'general_is_script',
        # Add 'general_is_archive', 'general_is_office', etc. as needed
        *[f'general_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved general

        # --- String Features ---
        'string_suspicious_hits', 'string_url_count', 'string_ip_count', 'string_base64_count',
        'string_pdb_count', 'string_user_agent_count', 'string_longest_len',
        'string_powershell_encoded_count', 'string_shadow_copy_delete_count', 'string_exploit_keyword_count',
        'string_malware_framework_count', 'string_c2_pattern_count',
        *[f'string_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved string

        # --- YARA Features ---
        *[f'yara_hit_{name}' for name in yara_rule_names], # Dynamic based on loaded rules
        'yara_total_hits',
        # No reserved needed here if list is dynamic, or add fixed number if preferred

        # --- Executable Specific (General) ---
        'exec_entrypoint_mean', 'exec_entrypoint_opcode_entropy',
        *[f'exec_entrypoint_opcode_hist_{i}' for i in range(TOP_N_OPCODES)],
        *[f'exec_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved exec-generic

        # --- PE Specific Features ---
        'pe_sizeof_image', 'pe_checksum', 'pe_sizeof_code', 'pe_machine', 'pe_dll_characteristics',
        'pe_linker_major_version', 'pe_linker_minor_version', 'pe_timestamp_diff_years', 'pe_timestamp_known_bad',
        'pe_num_tls_callbacks', 'pe_overlay_size', 'pe_overlay_entropy', 'pe_imphash', 'pe_exphash',
        'pe_num_imports', 'pe_num_suspicious_imports', 'pe_imports_only_ordinals', 'pe_num_exports',
        'pe_has_com_exports', 'pe_num_resource_langs', 'pe_has_english_lang',
        'pe_has_suspicious_lang', 'pe_num_resources', 'pe_manifest_uac', 'pe_num_relocations',
        'pe_entrypoint_suspicious', 'pe_entrypoint_rwx_r', 'pe_entrypoint_rwx_w', 'pe_entrypoint_rwx_x',
        'pe_num_sections', 'pe_section_avg_entropy', 'pe_section_low_entropy_present', 'pe_section_high_entropy_present',
        *[f'pe_section_ratio_{i}' for i in range(MAX_SECTION_RATIOS)],
        'pe_zero_size_sections', 'pe_executable_writable_sections', 'pe_is_dotnet', 'pe_rich_version_hash',
        'pe_rich_tooling_hash', 'pe_rich_num_tools', 'pe_rich_num_unique_tools', 'pe_rich_entropy',
        'pe_is_signed', 'pe_signature_expired',
        *[f'pe_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved PE

        # --- ELF Specific Features ---
        'elf_entrypoint', 'elf_num_segments', 'elf_num_dynamic_symbols', 'elf_num_sections',
        'elf_num_libraries', 'elf_has_nx', 'elf_has_pie', 'elf_has_rpath', 'elf_has_interpreter',
        *[f'elf_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved ELF

        # --- Mach-O Specific Features ---
        'macho_entrypoint', 'macho_num_commands', 'macho_num_libraries', 'macho_num_sections',
        'macho_has_code_signature', 'macho_has_objc', 'macho_has_encryption_info',
        *[f'macho_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved MachO

        # --- PDF Specific Features ---
        'pdf_page_count', 'pdf_is_encrypted', 'pdf_obj_count', 'pdf_stream_count', 'pdf_image_count',
        'pdf_font_count', 'pdf_embedded_file_count', 'pdf_javascript_present', 'pdf_launch_action_present',
        'pdf_openaction_present', 'pdf_uri_count', 'pdf_avg_stream_entropy', 'pdf_metadata_hash',
        *[f'pdf_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved PDF

        # --- Script Specific Features ---
        'script_line_count', 'script_avg_line_length', 'script_max_line_length', 'script_keyword_eval_count',
        'script_keyword_exec_count', 'script_keyword_http_count', 'script_keyword_socket_count',
        'script_obfuscation_indicator_longlines', 'script_obfuscation_indicator_entropy',
        *[f'script_reserved_{i}' for i in range(1, RESERVED_PER_BLOCK + 1)], # Reserved Script

        # --- Add blocks for other types (Archive, Office, Image, etc.) here ---
        # 'archive_...', etc.

        # --- Optional Overall Reserved Block at the very end ---
        # *[f'overall_reserved_{i}' for i in range(1, 21)], # Example: 20 overall reserved slots
    ]
    return order

FINAL_FEATURE_ORDER = define_feature_order(SORTED_YARA_RULE_NAMES)
FINAL_VECTOR_SIZE = len(FINAL_FEATURE_ORDER)
# FEATURE_NAME_TO_INDEX mapping is not strictly needed if assembling via loop, but can be useful
FEATURE_NAME_TO_INDEX = {name: idx for idx, name in enumerate(FINAL_FEATURE_ORDER)}
logger.info(f"Defined final feature vector order with {FINAL_VECTOR_SIZE} features.")

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper

@timer
def load_suspicious_terms(filepath):
    """Loads suspicious terms from a file, one per line."""
    terms = set()
    if not os.path.exists(filepath):
        logger.warning(f"Suspicious strings file not found at: {filepath}. String feature 'string_hits' will be 0.")
        return terms
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                term = line.strip().lower()
                if term:
                    terms.add(term)
        logger.info(f"Loaded {len(terms)} suspicious terms from {filepath}")
    except Exception as e:
        logger.error(f"Failed to read suspicious terms from '{filepath}': {e}")
    return terms

# --- Load Suspicious Terms ---
SUSPICIOUS_TERMS_SET = load_suspicious_terms(SUSPICIOUS_STRINGS_PATH) # Already loaded, just ensure it's defined before use

@timer
def entropy(data):
    if isinstance(data, memoryview):
        # logger.debug("Entropy input is memoryview, converting to bytes.") # Optional: remove debug noise
        data = bytes(data)
    """Calculates Shannon entropy for byte data."""
    if not data:
        return 0.0
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return 0.0
        # Ensure numpy array is treated as bytes if necessary
        if data.dtype != np.uint8:
            try:
                 data = data.astype(np.uint8).tobytes()
            except ValueError:
                 logger.warning(f"Cannot convert numpy array of type {data.dtype} to bytes for entropy calc.")
                 return 0.0

    if isinstance(data, str):
         logger.warning("Entropy input is string, encoding to utf-8 for calculation.")
         data = data.encode('utf-8', errors='ignore')

    if not isinstance(data, (bytes, bytearray)):
         logger.warning(f"Entropy input type {type(data)} not bytes/bytearray. Returning 0.0.")
         return 0.0


    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    total_bytes = len(data)
    if total_bytes == 0:
        return 0.0

    probs = byte_counts[byte_counts > 0] / total_bytes
    return -np.sum(probs * np.log2(probs))

@timer
def load_yara_rules_from_directory(rule_dir):
    rules = {}
    for filename in os.listdir(rule_dir):
        if filename.endswith(".yar") or filename.endswith(".yara"):
            full_path = os.path.join(rule_dir, filename)
            namespace = os.path.splitext(filename)[0]
            rules[namespace] = full_path
    return yara.compile(filepaths=rules) if rules else None

@timer
def run_yara_rules(raw_bytes, compiled_rules, filepath_for_logging="<memory>"):
    """Matches compiled YARA rules against file content bytes."""
    rule_hits = {} # Dictionary to store rule_name: hit (0 or 1)
    if not compiled_rules or not raw_bytes: # Added check for raw_bytes
        logger.warning(f"Skipping YARA scan for {filepath_for_logging}: No rules or empty content.")
        return rule_hits
    try:
        matches = compiled_rules.match(data=raw_bytes)
        # Convert matches to a set of hit rule names for quick lookup
        hit_names_set = {match.rule for match in matches}
        # Populate the dictionary based on ALL compiled rules
        for rule in compiled_rules:
            rule_name = rule.identifier
            rule_hits[rule_name] = 1.0 if rule_name in hit_names_set else 0.0
    except yara.Error as e:
        # Log error with file path for context, use repr(e) for more detail potentially
        logger.error(f"YARA match error on data from {filepath_for_logging}: {repr(e)}")
        # Return empty dict, main orchestrator will handle defaults
    except Exception as e:
        logger.error(f"Unexpected error during YARA scan for {filepath_for_logging}: {e}", exc_info=True) # Added exc_info

    return rule_hits

@timer
def safe_str_hash(text, modulus=10007):
    """Simple hash function for strings."""
    if not text or not isinstance(text, str):
        return 0
    # Use Python's built-in hash for simplicity and better distribution
    # Make it stable across runs and positive
    return abs(hash(text)) % modulus

@timer
def count_resources(resource_node):
     """Recursively counts resource nodes (directories and data)."""
     if not resource_node:
          return 0
     count = 0
     if hasattr(resource_node, 'childs'): # It's a directory node
          count = len(resource_node.childs) # Count immediate children
          for child in resource_node.childs:
               count += count_resources(child) # Recurse
     elif isinstance(resource_node, lief.PE.ResourceData): # It's a data node
          count = 1 # Count the data node itself
     return count

# --- Executable Generic Feature Functions ---

@timer
def extract_entrypoint_bytes_mean(binary, byte_count=1024):
    """
    Extracts bytes from executable entry point (PE/ELF/MachO) using LIEF
    and calculates their mean. Returns (bytes, mean).
    Handles PE/ELF/MachO differences in entrypoint access.
    """
    mean = 0.0
    ep_bytes = b'' # Default to empty bytes
    try:
        # LIEF abstracts entrypoint access
        ep = binary.entrypoint
        if ep == 0:
             logger.warning(f"Entry point address is 0 for {type(binary)}. Cannot read entrypoint bytes.")
             return ep_bytes, mean

        # get_content_from_virtual_address works for PE, ELF, MachO
        ep_bytes_list = binary.get_content_from_virtual_address(ep, byte_count)

        if ep_bytes_list:
            ep_bytes = bytes(ep_bytes_list) # Convert list of ints to bytes
            if ep_bytes:
                mean = np.mean(np.frombuffer(ep_bytes, dtype=np.uint8))
                logger.debug(f"Read {len(ep_bytes)} bytes from entry point 0x{ep:X}. Mean: {mean:.4f}")
            else:
                 # This case should be rare if ep_bytes_list was not empty
                 logger.warning(f"Conversion to bytes resulted in empty data for entry point 0x{ep:X}.")
                 mean = 0.0
        else:
            logger.warning(f"Could not read bytes from entry point VA 0x{ep:X}. Using default mean 0.0.")
            mean = 0.0

    except AttributeError:
         logger.warning(f"Binary object {type(binary)} lacks expected attributes for entry point extraction.")
         mean = 0.0
    except Exception as e:
        logger.error(f"Unexpected error extracting entry point bytes: {e}", exc_info=True)
        mean = 0.0

    # Ensure return types are consistent
    return ep_bytes if isinstance(ep_bytes, bytes) else b'', float(mean)

@timer
def extract_opcode_features_py(byte_code, top_n=TOP_N_OPCODES, arch=capstone.CS_ARCH_X86, mode=capstone.CS_MODE_32):
    """
    Calculates opcode entropy and top N histogram using Capstone.
    Accepts optional arch and mode for Capstone initializer. # Added documentation
    Returns (entropy, histogram_list).
    """
    opcode_entropy = 0.0
    histogram = [0.0] * top_n # Initialize histogram with zeros

    if not byte_code:
        # logger.warning("Input bytes for opcode extraction are empty. Returning zero features.") # Optional: uncomment if needed
        return opcode_entropy, histogram

    try:
        # Use the passed arch and mode arguments to initialize Capstone
        logger.debug(f"Initializing Capstone with arch={arch} mode={mode}") # Log arch/mode being used
        md = capstone.Cs(arch, mode)
        md.detail = False # Don't need detailed instruction info

        instructions = list(md.disasm(byte_code, 0x0)) # Disassemble from virtual address 0

        if not instructions:
            logger.warning("No instructions disassembled from input bytes. Returning zero opcode features.")
            return opcode_entropy, histogram

        mnemonics = [insn.mnemonic for insn in instructions]

        # Calculate frequency
        freq = defaultdict(int)
        for op in mnemonics:
            freq[op] += 1

        total = len(mnemonics)
        if total == 0:
             # This case should be rare if 'instructions' was not empty, but good to check
             logger.warning("Zero total valid opcodes found after disassembly. Returning zero features.")
             return 0.0, [0.0] * top_n

        # Calculate entropy
        # Use probabilities only for opcodes that actually occurred (count > 0)
        probs = [count / total for count in freq.values()] # No need to filter for count > 0 here
        opcode_entropy = -sum(p * np.log2(p) for p in probs if p > 0) # Filter p > 0 here for log2

        # Calculate top N histogram
        # Sort by frequency (descending)
        sorted_freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)
        # Get frequencies (count / total) for the top N opcodes
        top_n_freq = [count / total for _, count in sorted_freq[:top_n]]
        # Fill the histogram list (already initialized with zeros)
        histogram[:len(top_n_freq)] = top_n_freq # Assign calculated frequencies

        logger.debug(f"Opcode features extracted: entropy={opcode_entropy:.4f}, histogram_len={len(histogram)}")

    except capstone.CsError as e:
        logger.error(f"Capstone disassembly failed (arch={arch}, mode={mode}): {e}")
        # Return default zero features on Capstone error
        opcode_entropy = 0.0
        histogram = [0.0] * top_n
    except Exception as e:
        logger.error(f"Unexpected error during opcode feature extraction (arch={arch}, mode={mode}): {e}", exc_info=True) # Added exc_info
        opcode_entropy = 0.0
        histogram = [0.0] * top_n

    # Ensure float return type for entropy
    return float(opcode_entropy), histogram


@timer
def extract_string_features_py(content: bytes, max_scan_bytes=MAX_STRING_SCAN, suspicious_terms=SUSPICIOUS_TERMS_SET):
    """
    Extracts string features similar to the Rust implementation.
    """
    string_features_dict = {}

    try:
        if not content:
            logger.warning(f"Content is empty. Returning zero string features.")
            return string_features_dict

        # --- Initialize counters ---
        string_hits = 0
        url_count = 0
        ip_count = 0
        base64_like = 0
        pdb_count = 0
        user_agent_count = 0
        longest_string = 0
        powershell_encoded = 0
        shadow_copy_deletion = 0
        exploit_target_keywords = 0
        malware_frameworks = 0
        c2_patterns = 0
        # --- End counter initialization ---

        current_string_bytes = bytearray()
        extracted_strings = []

        # Extract printable ASCII strings (>= 5 chars)
        for byte in content:
            if 32 <= byte <= 126: # Printable ASCII range
                current_string_bytes.append(byte)
            else:
                if len(current_string_bytes) >= 5:
                    extracted_strings.append(bytes(current_string_bytes)) # Store as bytes
                    if len(current_string_bytes) > longest_string:
                        longest_string = len(current_string_bytes)
                current_string_bytes.clear()
        # Handle string at EOF
        if len(current_string_bytes) >= 5:
             extracted_strings.append(bytes(current_string_bytes))
             if len(current_string_bytes) > longest_string:
                 longest_string = len(current_string_bytes)

        logger.debug(f"Extracted {len(extracted_strings)} strings (>=5 chars) from first {len(content)} bytes.")

        # Analyze extracted strings
        for s_bytes in extracted_strings:
            try:
                # Decode carefully, ignoring errors, convert to lower
                s_lower = s_bytes.decode('ascii', errors='ignore').lower()
                if not s_lower: continue # Skip if decoding results in empty string

                # Check against suspicious terms set
                # Use intersection for efficiency if set is large, or simple loop
                hit_found = False
                for term in suspicious_terms:
                    if term in s_lower:
                        string_hits += 1
                        hit_found = True
                        break # Count first hit per string only

                # Other checks using regex or string methods
                if URL_REGEX.search(s_lower): url_count += 1
                if IP_REGEX.search(s_lower): ip_count += 1
                # Check base64-like - use regex or Rust's simpler check
                # if BASE64_LIKE_REGEX.match(s_lower): base64_like += 1
                # Rust's check: >= 16 chars, all alphanumeric or +/=
                if len(s_lower) >= 16 and all(c.isalnum() or c in '+/=' for c in s_lower): base64_like += 1
                if s_lower.endswith(".pdb"): pdb_count += 1
                if USER_AGENT_REGEX.search(s_lower): user_agent_count += 1
                if POWERSHELL_ENCODED_REGEX.search(s_lower): powershell_encoded += 1
                if VSSADMIN_REGEX.search(s_lower): shadow_copy_deletion += 1
                if EXPLOIT_TARGET_REGEX.search(s_lower): exploit_target_keywords += 1
                if MALWARE_FRAMEWORK_REGEX.search(s_lower): malware_frameworks += 1
                if C2_PATTERN_REGEX.search(s_lower): c2_patterns += 1

            except Exception as decode_err:
                logger.warning(f"Could not decode or process string chunk: {decode_err}")
                continue # Skip this string chunk if decoding fails badly


        string_features_dict['string_suspicious_hits'] = float(string_hits)
        string_features_dict['string_url_count'] = float(url_count)
        string_features_dict['string_ip_count'] = float(ip_count)
        string_features_dict['string_base64_count'] = float(base64_like) # Ensure base64_like counter exists
        string_features_dict['string_pdb_count'] = float(pdb_count)
        string_features_dict['string_user_agent_count'] = float(user_agent_count)
        string_features_dict['string_longest_len'] = float(longest_string)
        string_features_dict['string_powershell_encoded_count'] = float(powershell_encoded) # Ensure counter exists
        string_features_dict['string_shadow_copy_delete_count'] = float(shadow_copy_deletion) # Ensure counter exists
        string_features_dict['string_exploit_keyword_count'] = float(exploit_target_keywords) # Ensure counter exists
        string_features_dict['string_malware_framework_count'] = float(malware_frameworks) # Ensure counter exists
        string_features_dict['string_c2_pattern_count'] = float(c2_patterns) # Ensure counter exists

        logger.debug("String feature extraction completed.")

    except Exception as e:
        logger.error(f"Error during string feature extraction: {e}", exc_info=True)

    return string_features_dict

@timer
def check_relocation_table(binary):
    num_relocations = 0
    try:
        for relocation in binary.relocations:
            num_relocations += 1
            logger.debug(f"Relocation address: {relocation.address}, size: {relocation.size} bits")
    except AttributeError:
        # Handle cases where the relocations are not present (in some binary types)
        logger.debug("No relocations available for this binary.")
    logger.debug(f"Found {num_relocations} relocations.")
    return num_relocations

# Function to check entry point behavior
@timer
def check_entry_point(binary):
    entry_point = binary.entrypoint
    base_address = binary.optional_header.imagebase
    entry_point_diff = entry_point - base_address
    # Consider entry points far from the base address as suspicious
    if entry_point_diff > 0x10000:  # Example threshold for large offsets
        return True
    return False

@timer
def extract_imports_and_exports(binary):
    # Process imports
    if binary.has_imports:
        import_strings = []
        imports = [lib.name.lower() for lib in binary.imports]
        suspicious_libs = ['shell32', 'advapi32', 'ws2_32', 'wininet', 'urlmon', 'oleaut32']
        for lib in binary.imports:
            libname = lib.name.lower() if lib.name else ""
            for entry in lib.entries:
                funcname = entry.name if entry.name else f"{entry.ordinal}"
                import_strings.append(f"{libname}.{funcname}")
        imphash = safe_str_hash("".join(import_strings))

        num_imports = len(binary.imports)
        suspicious_count = sum(1 for lib in imports if any(s in lib for s in suspicious_libs))
        uses_only_ordinals = int(
            all(
                all(entry.is_ordinal for entry in lib.entries)
                for lib in binary.imports
            )
        )
        logger.debug(f"imphash: {imphash}, num_imports: {num_imports}, suspicious_count: {suspicious_count}, uses_only_ordinals: {uses_only_ordinals}")
    else:
        logger.debug("No imports found")
        imphash = 0
        num_imports = 0
        suspicious_count = 0
        uses_only_ordinals = 0

    # --- Process exports ---
    exphash = 0
    export_count = 0
    com_export = 0
    if hasattr(binary, 'has_exports') and binary.has_exports: # Add hasattr check
        export_names = []
        export_access_errors = 0
        num_exported_funcs = 0
        if hasattr(binary, 'exported_functions'): # Check if attribute exists
             num_exported_funcs = len(binary.exported_functions)
             for exp in binary.exported_functions:
                 try:
                     # --- Add try/except around name access ---
                     exp_name = None
                     if hasattr(exp, 'name') and exp.name: # Check attribute exists first
                         exp_name_raw = exp.name
                         if isinstance(exp_name_raw, bytes):
                             exp_name = exp_name_raw.decode('utf-8', errors='ignore')
                         else:
                             exp_name = str(exp_name_raw) # Ensure string

                     if exp_name: # Only append if name resolution worked
                         export_names.append(exp_name)
                     # --- End try/except ---
                 except Exception as e_exp:
                     # Log that accessing this specific export name failed
                     # This might correlate with the "Can't read" errors from LIEF C++
                     ordinal = getattr(exp, 'ordinal', 'N/A') # Get ordinal if possible
                     address = getattr(exp, 'address', 'N/A') # Get address if possible
                     logger.warning(f"Failed to get/decode name for export entry (ordinal: {ordinal}, address: {hex(address) if isinstance(address, int) else address}): {e}. LIEF internal errors might be occurring.")
                     export_access_errors += 1
        else:
             logger.warning("'exported_functions' attribute not found on binary object.")


        if export_names: # Calculate hash only if names were found
             exphash = safe_str_hash("".join(export_names))
             com_export = int(any(n.lower() in ['dllmain', 'dllregisterserver'] for n in export_names))

        # Report total attempted vs successfully named exports
        export_count = len(export_names)
        logger.debug(f"Exports: total_funcs={num_exported_funcs}, named_funcs={export_count}, name_errors={export_access_errors}, com_export={com_export}, exphash={exphash}")
        if export_access_errors > 0:
             logger.warning(f"{export_access_errors} export names could not be resolved. Export hash may be inaccurate.")

    else:
        logger.debug("No exports found or 'has_exports' is false.")
        # Keep defaults: exphash=0, export_count=0, com_export=0

    return imphash, exphash, num_imports, suspicious_count, uses_only_ordinals, export_count, com_export

@timer
def extract_language_ids(binary):
    """
    Extracts language IDs from the resources section of a PE binary.
    """
    language_ids = []

    if not binary.has_resources:
        return language_ids

    root = binary.resources

    for resource_type in root.childs:
        if isinstance(resource_type, lief.PE.ResourceDirectory):
            try:
                # Convert the resource type to integer and compare with MANIFEST
                if int(resource_type.id) == lief.PE.ResourcesManager.TYPE.MANIFEST:
                    for id_node in resource_type.childs:
                        if isinstance(id_node, lief.PE.ResourceDirectory):
                            for lang_node in id_node.childs:
                                if isinstance(lang_node, lief.PE.ResourceData):
                                    language_ids.append(lang_node.id)
            except Exception as e:
                continue  # Skip to the next resource if there is an error

    return language_ids

@timer
def extract_opcode_references(content):
    logger.debug("Performing opcode reference extraction ...")
    address_count = defaultdict(int)
    INSTRUCTION_BYTES_TO_CHECK = {0xE8, 0xE9}  # LEA, MOV, CALL, JMP
    i = 0
    while i < len(content) - 5:
        op = content[i]
        if op in INSTRUCTION_BYTES_TO_CHECK:
            # Skip 1 byte for opcode, and grab next 4 bytes as operand
            if i + 5 <= len(content):
                operand_bytes = content[i+1:i+5]
                operand_val = int.from_bytes(operand_bytes, byteorder='little')
                address_count[operand_val] += 1
                i += 5  # Move past the operand
                continue
        i += 1  # Move to next byte
    return address_count

# Generate feature vector from the opcode reference counts
@timer
def generate_opcode_feature_vector(address_count):
    # Prepare address frequency data
    address_frequencies = np.array(list(address_count.values()))
    
    if len(address_frequencies) == 0:
        return [0.0] * 5  # Return a vector of zeros if no references found

    # Mean distribution of addresses
    mean_address = np.mean(address_frequencies)
    logger.debug(f"Mean address: {mean_address}")
    
    # Highest frequency (most referenced address)
    max_freq = np.max(address_frequencies)
    logger.debug(f"Max frequency: {max_freq}")
    
    # Total number of unique addresses
    unique_addresses = len(address_frequencies)
    logger.debug(f"Unique addresses: {unique_addresses}")
    
    # Max Frequency (the count of the most frequently referenced address)
    highest_freq_address = list(address_count.keys())[np.argmax(address_frequencies)]
    logger.debug(f"Highest frequency address: 0x{highest_freq_address:X}")
    
    # Entropy of address frequency distribution
    address_entropy = entropy(address_frequencies)
    logger.debug(f"Address entropy: {address_entropy}")

    # Construct feature vector (mean, max_freq, unique addresses, highest freq address, address entropy)
    feature_vector = [
        float(mean_address),
        float(max_freq),
        float(unique_addresses),
        float(highest_freq_address),
        float(address_entropy)
    ]
    
    return feature_vector

@timer
def find_entry_section(binary):
    # Get the entry point of the binary (Relative Virtual Address)
    ep = binary.entrypoint
    image_base = binary.optional_header.imagebase
    logger.debug(f"Entrypoint (RVA): 0x{ep:X}, Image Base: 0x{image_base:X}")
    
    # Debug: Log each section's virtual address and virtual size (adjusted with Image Base)
    for s in binary.sections:
        section_va = s.virtual_address + image_base  # Adjust section VA with Image Base
        logger.debug(f"Section {s.name}: VA=0x{section_va:X}, Virtual Size=0x{s.virtual_size:X}, Raw Size=0x{s.size:X}")
    
    # Try to find the section that contains the entry point
    entry_section = next(
        (s for s in binary.sections if s.virtual_address + image_base <= ep < s.virtual_address + s.virtual_size + image_base),
        None
    )

    if entry_section:
        logger.debug(f"Found entry section: {entry_section.name} (VA: 0x{entry_section.virtual_address + image_base:X}, Size: 0x{entry_section.virtual_size:X})")
    return entry_section

@timer
def extract_features_pe(path, raw_bytes, binary):
    """
    Extracts PE-specific features using LIEF and returns them as a dictionary.
    Generic features (strings, overall entropy, yara, exec opcodes) are handled elsewhere.
    """
    logger.debug(f"Extracting PE features for {path} (Size: {len(raw_bytes)} bytes)...")
    features = {} # Initialize dictionary for PE features
    start_time = time.time()

    try:
        # --- PE Header Info ---
        oh = binary.optional_header
        hdr = binary.header
        features['pe_sizeof_image'] = float(oh.sizeof_image)
        features['pe_checksum'] = float(oh.checksum)
        features['pe_sizeof_code'] = float(oh.sizeof_code)
        features['pe_machine'] = float(hdr.machine.value)
        features['pe_dll_characteristics'] = float(oh.dll_characteristics)
        features['pe_linker_major_version'] = float(oh.major_linker_version)
        features['pe_linker_minor_version'] = float(oh.minor_linker_version)

        # --- Timestamp ---
        timestamp_year = datetime.fromtimestamp(hdr.time_date_stamps, timezone.utc).year if hdr.time_date_stamps else 1970
        current_year = datetime.now(timezone.utc).year
        features['pe_timestamp_diff_years'] = float(current_year - timestamp_year)
        features['pe_timestamp_known_bad'] = float(timestamp_year in {1992, 2007, 2040}) # Example known bad years

        # --- TLS ---
        features['pe_num_tls_callbacks'] = float(len(binary.tls.callbacks) if binary.has_tls and binary.tls.callbacks else 0)

        # --- Overlay ---
        file_size = len(raw_bytes)
        overlay_size = 0.0
        overlay_entropy = 0.0
        if binary.sections:
            try: # Handle potential errors with section properties
                last_section = max(binary.sections, key=lambda s: s.offset + s.size)
                expected_end = last_section.offset + last_section.size
                overlay_size = max(file_size - expected_end, 0)
                if overlay_size > 0:
                    overlay_data = raw_bytes[-int(overlay_size):] # Slice requires int
                    overlay_entropy = entropy(overlay_data)
            except Exception as ov_err:
                 logger.warning(f"Could not calculate overlay for {path}: {ov_err}")
        features['pe_overlay_size'] = float(overlay_size)
        features['pe_overlay_entropy'] = float(overlay_entropy)

        # --- Imports / Exports ---
        imphash, exphash, num_imports, suspicious_count, uses_only_ordinals, export_count, com_export = extract_imports_and_exports(binary)
        features['pe_imphash'] = float(imphash)
        features['pe_exphash'] = float(exphash)
        features['pe_num_imports'] = float(num_imports)
        features['pe_num_suspicious_imports'] = float(suspicious_count) # Count of imports from suspicious DLLs
        features['pe_imports_only_ordinals'] = float(uses_only_ordinals)
        features['pe_num_exports'] = float(export_count)
        features['pe_has_com_exports'] = float(com_export) # Exports like DllMain/DllRegisterServer

        # --- Resources ---
        language_ids = extract_language_ids(binary) # Assumes this returns list of lang IDs
        features['pe_num_resource_langs'] = float(len(language_ids))
        features['pe_has_english_lang'] = float(0x0409 in language_ids) # Check if English (US) is present
        features['pe_has_suspicious_lang'] = float(any(lang_id not in {0x0409, 0x0000} for lang_id in language_ids)) # Check for non-English/neutral
        features['pe_num_resources'] = float(count_resources(binary.resources) if binary.has_resources else 0)
        manifest_uac = 0
        if binary.has_resources and hasattr(binary, 'resources_manager'): # Check for resources_manager
            try:
                # Access manifest carefully, it might not exist or be parsable
                 manifest_str = binary.resources_manager.manifest or ""
                 manifest_uac = float("autoElevate" in manifest_str or "requireAdministrator" in manifest_str)
            except Exception as e:
                 logger.warning(f"Error processing manifest UAC for {path}: {e}")
                 manifest_uac = 0.0 # Default if error
        features['pe_manifest_uac'] = float(manifest_uac)


        # --- Relocations ---
        features['pe_num_relocations'] = float(check_relocation_table(binary))

        # --- Entry Point ---
        features['pe_entrypoint_suspicious'] = float(check_entry_point(binary)) # Check if EP is far from image base

        # --- Entry Point RWX flags ---
        # NOTE: This ALSO relies on section characteristics check below. If that fails, these will be 0.
        entry_section = find_entry_section(binary)
        rwx_flags = [0.0, 0.0, 0.0]
        if entry_section:
            try:
                # --- Use RAW VALUES for characteristics check here too ---
                MEM_EXECUTE = 0x20000000
                MEM_READ =    0x40000000
                MEM_WRITE =   0x80000000
                if hasattr(entry_section, 'characteristics') and isinstance(entry_section.characteristics, int):
                     has_read = bool(entry_section.characteristics & MEM_READ)
                     has_write = bool(entry_section.characteristics & MEM_WRITE)
                     has_exec = bool(entry_section.characteristics & MEM_EXECUTE)
                     rwx_flags = [float(has_read), float(has_write), float(has_exec)]
                     logger.debug(f"Entry section '{entry_section.name}' RWX flags: R={rwx_flags[0]} W={rwx_flags[1]} X={rwx_flags[2]}")
                else:
                     logger.warning(f"Entry section '{entry_section.name}' characteristics not accessible. Cannot determine RWX.")
            except Exception as entry_char_err:
                 logger.warning(f"Error checking entry section characteristics: {entry_char_err}")
        else:
            logger.warning(f"Could not find entry point section for {path}. Cannot determine RWX.") # Log if section not found
        features['pe_entrypoint_rwx_r'] = rwx_flags[0]
        features['pe_entrypoint_rwx_w'] = rwx_flags[1]
        features['pe_entrypoint_rwx_x'] = rwx_flags[2]

        # --- Sections ---
        num_sections = len(binary.sections)
        features['pe_num_sections'] = float(num_sections)
        section_entropies = [] # List to store entropy of sections with content
        section_ratios = []    # List to store physical/virtual size ratio
        zero_size_sections = 0 # Counter for sections with physical size 0
        executable_writable_sections = 0 # Counter for sections with W+X flags

        if num_sections > 0:
            # Iterate through all sections
            for s in binary.sections:
                # Calculate entropy and size ratio for sections with content
                if s.size > 0:
                    # Ensure content is bytes for entropy calculation
                    # section.content can be memoryview, convert it
                    section_bytes = bytes(s.content)
                    section_entropies.append(entropy(section_bytes)) # Pass bytes

                    # Calculate physical vs virtual size ratio, handle division by zero
                    ratio = float(s.size / s.virtual_size if s.virtual_size > 0 else 0)
                    section_ratios.append(ratio)
                else:
                    # Section has no physical content
                    zero_size_sections += 1
                    section_ratios.append(0.0) # Ratio is 0

                try:
                    # Standard PE Section Characteristic Values
                    MEM_EXECUTE = 0x20000000
                    MEM_WRITE =   0x80000000

                    is_write = False
                    is_exec = False
                    # Check if characteristics attribute exists and is integer-like
                    if hasattr(s, 'characteristics') and isinstance(s.characteristics, int):
                         # Perform bitwise AND directly with integer constants
                         is_write = bool(s.characteristics & MEM_WRITE)
                         is_exec = bool(s.characteristics & MEM_EXECUTE)
                         if is_write and is_exec:
                              executable_writable_sections += 1
                    else:
                         # Log warning if characteristics cannot be checked for this section
                         logger.warning(f"Section '{s.name}' characteristics not accessible or not integer. Cannot check W+X.")

                except Exception as char_err:
                     # Catch any unexpected error during characteristic check
                     logger.warning(f"Error checking characteristics for section '{s.name}': {char_err}")

            # --- Post-loop processing of collected section data ---

            # Pad or truncate section ratios to ensure fixed length (MAX_SECTION_RATIOS)
            # Append zeros if fewer sections than MAX_SECTION_RATIOS
            # Slice if more sections than MAX_SECTION_RATIOS
            padded_section_ratios = (section_ratios + [0.0] * MAX_SECTION_RATIOS)[:MAX_SECTION_RATIOS]
            # Assign padded ratios to feature dictionary
            for i, ratio in enumerate(padded_section_ratios):
                features[f'pe_section_ratio_{i}'] = float(ratio)

            # Calculate entropy statistics only if there were sections with content
            if section_entropies: # Check if the list is not empty
                features['pe_section_avg_entropy'] = float(np.mean(section_entropies))
                # Feature: Presence of any section with very low entropy (< 1.0)
                features['pe_section_low_entropy_present'] = float(any(e < 1.0 for e in section_entropies))
                # Feature: Presence of any section with very high entropy (> 7.5, often packed/encrypted)
                features['pe_section_high_entropy_present'] = float(any(e > 7.5 for e in section_entropies))
            else:
                # Default values if no sections had content
                features['pe_section_avg_entropy'] = 0.0
                features['pe_section_low_entropy_present'] = 0.0
                features['pe_section_high_entropy_present'] = 0.0

        else: # Case where the PE file has no sections listed
            logger.warning(f"No sections found in PE file {path}. Setting section features to defaults.")
            # Set default values for all section-related features
            features['pe_section_avg_entropy'] = 0.0
            features['pe_section_low_entropy_present'] = 0.0
            features['pe_section_high_entropy_present'] = 0.0
            # Ensure all ratio features are initialized
            for i in range(MAX_SECTION_RATIOS):
                features[f'pe_section_ratio_{i}'] = 0.0
            # Counters are already 0

        # Add the final counts to the features dictionary
        features['pe_zero_size_sections'] = float(zero_size_sections)
        features['pe_executable_writable_sections'] = float(executable_writable_sections)

        # --- End of Sections block ---

        is_dotnet = binary.has_debug and any("CLR" in (s.name.decode('utf-8', errors='ignore') if isinstance(s.name, bytes) else s.name) for s in binary.sections)
        features['pe_is_dotnet'] = float(is_dotnet)

        # --- Rich Header ---
        rich_version = 0.0
        rich_tooling = 0.0
        num_rich_tools = 0.0
        unique_tool_ids = 0.0
        rich_entropy_val = 0.0 # Renamed from rich_entropy
        if binary.has_rich_header:
            try:
                entries = binary.rich_header.entries
                num_rich_tools = len(entries)
                if num_rich_tools > 0:
                    entry_ids = [e.id for e in entries]
                    unique_tool_ids = len(set(entry_ids))
                    rich_version = sum(entry_ids) % 10007 # Hash of IDs
                    rich_tooling = safe_str_hash("".join(f"{e.id}:{e.build_id}" for e in entries)) # Use build_id too
                    # Calculate rich_entropy using id and build_id bytes
                    rich_entropy_data = b"".join(e.id.to_bytes(2, 'little') + e.build_id.to_bytes(2, 'little') for e in entries)
                    rich_entropy_val = entropy(rich_entropy_data)

            except Exception as e:
                logger.error(f"Error processing RICH header for {path}: {e}")
                # Defaults are already 0.0
        features['pe_rich_version_hash'] = float(rich_version)
        features['pe_rich_tooling_hash'] = float(rich_tooling)
        features['pe_rich_num_tools'] = float(num_rich_tools)
        features['pe_rich_num_unique_tools'] = float(unique_tool_ids)
        features['pe_rich_entropy'] = float(rich_entropy_val)

        # --- Signature ---
        # LIEF's has_signatures is basic. A real check needs external libraries (like pyasn1, cryptography)
        # or dedicated tools. Setting expired to 0 as placeholder.
        features['pe_is_signed'] = float(binary.has_signatures)
        features['pe_signature_expired'] = 0.0 # Placeholder

        logger.debug(f"PE Feature Dict Extraction {path} completed in {time.time() - start_time:.2f}s. Found {len(features)} PE features.")
        return features # <<< Return the dictionary

    except Exception as e:
        logger.error(f"Error extracting PE features for {path}: {e}")
        logger.error(traceback.format_exc())
        return {} # Return empty dict on other errors

@timer
def extract_elf_features(path, raw_bytes, binary): # Added path for logging
    """Extracts ELF-specific features."""
    logger.info(f"Extracting ELF features for {path}...")
    features = {}
    try:
        features['elf_entrypoint'] = float(binary.entrypoint)
        features['elf_num_segments'] = float(len(binary.segments))
        features['elf_num_dynamic_symbols'] = float(len(binary.dynamic_symbols))
        features['elf_num_sections'] = float(len(binary.sections))
        features['elf_num_libraries'] = float(len(binary.libraries))
        features['elf_has_nx'] = float(binary.has_nx)
        features['elf_has_pie'] = float(binary.has_pie)
        features['elf_has_rpath'] = float(binary.has_rpath)
        features['elf_has_interpreter'] = float(binary.has_interpreter)
        # TODO: Add more ELF features (e.g., section flags, program header flags)
    except Exception as e:
        logger.error(f"Error processing ELF {path}: {e}", exc_info=True)
        return {}
    return features

@timer
def extract_macho_features(path, raw_bytes, binary): # Added path for logging
    """Extracts Mach-O specific features."""
    logger.info(f"Extracting Mach-O features for {path}...")
    features = {}
    try:
        features['macho_entrypoint'] = float(binary.entrypoint)
        features['macho_num_commands'] = float(len(binary.commands))
        features['macho_num_libraries'] = float(len(binary.libraries))
        features['macho_num_sections'] = float(len(binary.sections))
        features['macho_has_code_signature'] = float(binary.has_code_signature)
        features['macho_has_objc'] = float(binary.has_objc)
        features['macho_has_encryption_info'] = float(binary.has_encryption_info)
        # TODO: Add more Mach-O features (e.g., load command details, entitlements)
    except Exception as e:
        logger.error(f"Error processing Mach-O {path}: {e}", exc_info=True)
        return {}
    return features

@timer
def extract_features_pdf(path, raw_bytes):
    """Extracts features from PDF files."""
    logger.info(f"Processing PDF {path}...")
    features = {}
    if not PDF_LIB_AVAILABLE:
        logger.warning("PDF processing skipped: PyPDF2 library not available.")
        return features

    try:
        reader = PyPDF2.PdfReader(path) # PyPDF2 needs path, not bytes usually
        metadata = reader.metadata
        num_pages = len(reader.pages)

        features['pdf_page_count'] = float(num_pages)
        features['pdf_is_encrypted'] = float(reader.is_encrypted)

        # Basic object counts (approximate with PyPDF2)
        # TODO: Use PyMuPDF (fitz) for more accurate counts of streams, images, fonts, embedded files
        features['pdf_obj_count'] = float(len(reader.objects)) # Indirect objects
        features['pdf_stream_count'] = 0.0 # Placeholder
        features['pdf_image_count'] = 0.0 # Placeholder
        features['pdf_font_count'] = 0.0 # Placeholder
        features['pdf_embedded_file_count'] = 0.0 # Placeholder

        # TODO: Implement checks for JS, Launch Actions, OpenAction, URIs by traversing objects/pages
        # PyMuPDF is generally better suited for this detailed analysis.
        features['pdf_javascript_present'] = 0.0 # Placeholder
        features['pdf_launch_action_present'] = 0.0 # Placeholder
        features['pdf_openaction_present'] = 0.0 # Placeholder
        features['pdf_uri_count'] = 0.0 # Placeholder

        # Metadata hash
        meta_text = ""
        if metadata:
            meta_text += metadata.author or ""
            meta_text += metadata.creator or ""
            meta_text += metadata.producer or ""
            meta_text += metadata.subject or ""
            meta_text += metadata.title or ""
        features['pdf_metadata_hash'] = float(safe_str_hash(meta_text))

        # TODO: Implement stream extraction and entropy calculation using PyMuPDF
        features['pdf_avg_stream_entropy'] = 0.0 # Placeholder

    except PyPDF2.errors.PdfReadError as e:
         logger.warning(f"PyPDF2 could not read PDF {path}: {e}")
         return {}
    except Exception as e:
        logger.error(f"Error processing PDF {path}: {e}", exc_info=True)
        return {}
    return features

@timer
def extract_features_script(path, raw_bytes):
    """Extracts features from generic script files."""
    logger.info(f"Processing Script {path}...")
    features = {}
    script_content = "" # Initialize
    try:
        # Try decoding with common encodings
        decoded = False
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings_to_try:
            try:
                script_content = raw_bytes.decode(enc)
                logger.debug(f"Decoded script {path} using {enc}")
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        if not decoded:
             # Fallback: Decode ignoring errors - may lose info but better than failing
             script_content = raw_bytes.decode('utf-8', errors='ignore')
             logger.warning(f"Could not decode script {path} cleanly, used fallback.")

        lines = script_content.splitlines()
        num_lines = len(lines)
        features['script_line_count'] = float(num_lines)
        if num_lines > 0:
             line_lengths = [len(line) for line in lines]
             features['script_avg_line_length'] = float(np.mean(line_lengths))
             features['script_max_line_length'] = float(np.max(line_lengths))
             features['script_obfuscation_indicator_longlines'] = float(np.max(line_lengths) > 1000) # Example threshold
        else:
             features['script_avg_line_length'] = 0.0
             features['script_max_line_length'] = 0.0
             features['script_obfuscation_indicator_longlines'] = 0.0

        # Keyword counts (case-insensitive, use word boundaries for less false positives)
        script_lower = script_content.lower()
        features['script_keyword_eval_count'] = float(len(re.findall(r'\beval\b', script_lower)))
        features['script_keyword_exec_count'] = float(len(re.findall(r'\bexec\b', script_lower))) # Python exec
        features['script_keyword_http_count'] = float(len(re.findall(r'http[s]?://', script_lower)))
        features['script_keyword_socket_count'] = float(len(re.findall(r'\bsocket\b', script_lower)))
        # TODO: Add more script-language specific keywords (e.g., CreateObject for VBS, require for JS/Node)

        # Entropy based obfuscation (use raw bytes entropy calculated earlier)
        # Compare general entropy to typical text entropy (~4.5 for English)
        general_entropy = entropy(raw_bytes) # Re-calc or pass from main? Pass is better. Assuming passed as arg eventually.
        features['script_obfuscation_indicator_entropy'] = float(general_entropy > 6.0) # Example threshold for high entropy script

        # TODO: Consider AST parsing for Python/JS for deeper analysis
        # TODO: Consider character frequency analysis

    except Exception as e:
        logger.error(f"Error processing script {path}: {e}", exc_info=True)
        return {}
    return features

@timer
def extract_features(path):
    """
    Main feature extraction orchestrator. Detects file type, calls appropriate
    extractors, and assembles the final unified feature vector.
    Includes temporary LIEF debugging prints.
    """
    all_features = {} # Dictionary to hold all collected features
    # Ensure initialization uses the final target size for consistency on errors
    output_vector = [0.0] * TARGET_VECTOR_SIZE
    binary = None # Initialize binary object variable
    is_pe = is_elf = is_macho = is_pdf = is_script = False # Initialize flags

    try:
        # --- 0. Read File ---
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
            file_size = len(raw_bytes)
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            return [0.0] * TARGET_VECTOR_SIZE # Return default zero vector
        except OSError as e:
            logger.error(f"OS error reading file {path}: {e}")
            return [0.0] * TARGET_VECTOR_SIZE # Return default zero vector

        if file_size == 0:
            logger.warning(f"File {path} is empty.")
            return [0.0] * TARGET_VECTOR_SIZE # Return default zero vector

        all_features['general_file_size'] = float(file_size)

        # --- 1. Run Generic Extractors ---
        gen_entropy = entropy(raw_bytes) # Assumes entropy() is defined
        all_features['general_entropy'] = gen_entropy

        # Entropy buckets
        if file_size > 0:
            # Assumes np is imported as numpy
            freq = np.bincount(np.frombuffer(raw_bytes, dtype=np.uint8), minlength=256)
            total = file_size
            # Assumes MAX_ENTROPY_BUCKETS is defined
            for i in range(MAX_ENTROPY_BUCKETS):
                start = i * (256 // MAX_ENTROPY_BUCKETS)
                end = (i + 1) * (256 // MAX_ENTROPY_BUCKETS)
                bucket_sum = np.sum(freq[start:end])
                all_features[f'general_entropy_bucket_{i}'] = float(bucket_sum / total)
        else:
            for i in range(MAX_ENTROPY_BUCKETS):
                all_features[f'general_entropy_bucket_{i}'] = 0.0

        # String features
        # Assumes extract_string_features_py, MAX_STRING_SCAN, SUSPICIOUS_TERMS_SET defined
        string_features_dict = extract_string_features_py(raw_bytes[:MAX_STRING_SCAN], MAX_STRING_SCAN, SUSPICIOUS_TERMS_SET)
        all_features.update(string_features_dict)

        # YARA features
        # Assumes run_yara_rules, COMPILED_YARA_RULES defined
        yara_results = run_yara_rules(raw_bytes, COMPILED_YARA_RULES, filepath_for_logging=path)
        all_features['yara_total_hits'] = float(sum(yara_results.values()))
        # Assumes SORTED_YARA_RULE_NAMES, FEATURE_NAME_TO_INDEX defined
        for rule_name in SORTED_YARA_RULE_NAMES:
            if rule_name:
                feature_key = f'yara_hit_{rule_name}'
                if feature_key in FEATURE_NAME_TO_INDEX:
                    all_features[feature_key] = float(yara_results.get(rule_name, 0.0))


        # --- 2. File Type Detection (Magic) ---
        magic_desc = ""
        magic_hash = 0.0
        # Assumes 'magic' object/library is imported and configured
        try:
            magic_desc = magic.from_buffer(raw_bytes).lower()
            # Assumes safe_str_hash defined
            magic_hash = float(safe_str_hash(magic_desc))
            logger.debug(f"Magic description for {path}: {magic_desc}")
        except Exception as magic_err:
            logger.warning(f"python-magic failed for {path}: {magic_err}")
        all_features['general_magic_hash'] = magic_hash
        # Assumes 'general_magic_desc_hash' is the intended feature name
        all_features['general_magic_desc_hash'] = float(safe_str_hash(magic_desc))


        # --- 3. LIEF Parsing using io.BytesIO ---
        mem_file = None
        parsed_obj = None
        try:
            logger.debug(f"Attempting LIEF parsing from io.BytesIO for {path}")
            mem_file = io.BytesIO(raw_bytes)
            mem_file.seek(0)
            parsed_obj = lief.parse(mem_file) # Pass file-like object

            if parsed_obj is None:
                logger.warning(f"lief.parse(io.BytesIO) returned None for file: {path}.")
            elif isinstance(parsed_obj, lief.PE.Binary):
                logger.debug(f"lief.parse(io.BytesIO) successful. Identified as PE.")
                is_pe = True
                binary = parsed_obj # Assign the parsed object
            elif isinstance(parsed_obj, lief.ELF.Binary):
                logger.debug(f"lief.parse(io.BytesIO) successful. Identified as ELF.")
                is_elf = True
                binary = parsed_obj
            elif isinstance(parsed_obj, lief.MachO.Binary):
                logger.debug(f"lief.parse(io.BytesIO) successful. Identified as MachO.")
                is_macho = True
                binary = parsed_obj
            else:
                logger.info(f"LIEF parsed file {path} as type {type(parsed_obj)}, but no specific executable extractor for it.")
                # Fallback check based on magic_desc/extension
                if 'pdf' in magic_desc or path.lower().endswith('.pdf'): is_pdf = True
                elif 'text' in magic_desc or path.lower().endswith(('.py', '.js', '.vbs', '.sh', '.bat', '.ps1')): is_script = True

        # Catch Exceptions during LIEF parsing or type checking
        except Exception as e_parse:
            logger.warning(f"LIEF parse(io.BytesIO) or subsequent type check failed for file {path}: {type(e_parse).__name__}: {e_parse}. Checking non-exec types.")
            binary = None # Ensure binary is None on any parsing failure
            # Fallback check based on magic_desc/extension
            if 'pdf' in magic_desc or path.lower().endswith('.pdf'): is_pdf = True
            elif 'text' in magic_desc or path.lower().endswith(('.py', '.js', '.vbs', '.sh', '.bat', '.ps1')): is_script = True

        finally:
            # Explicitly close the BytesIO object
            if mem_file:
                mem_file.close()

        # Update type flags based on parsing results or fallbacks
        all_features['general_is_pe'] = float(is_pe)
        all_features['general_is_elf'] = float(is_elf)
        all_features['general_is_macho'] = float(is_macho)
        all_features['general_is_pdf'] = float(is_pdf)
        all_features['general_is_script'] = float(is_script)


        # --- 4. Run Specific Extractors Based on Type ---
        if is_pe and binary:
            logger.debug(f"Calling extract_features_pe for {path}")
            # Assumes extract_features_pe is defined
            pe_feats = extract_features_pe(path, raw_bytes, binary)
            all_features.update(pe_feats)

            # Extract Entrypoint/Opcodes (using default arch/mode for this debug run)
            logger.debug("Attempting PE entrypoint/opcode extraction (using default arch/mode for debug)...")
            try:
                entrypoint_bytes, byte_mean = extract_entrypoint_bytes_mean(binary, 1024)
                pe_arch = capstone.CS_ARCH_X86
                pe_mode = capstone.CS_MODE_32

                if hasattr(binary, 'header') and hasattr(binary.header, 'machine'):
                    machine_obj = binary.header.machine

                    # --- Compare machine type using integer value ---
                    if hasattr(machine_obj, 'value'):
                        machine_val = machine_obj.value
                        logger.debug(f"Machine type raw value: {machine_val}")
                        # Standard PE Machine Type Values
                        I386_VAL = 0x14c  # 332
                        AMD64_VAL = 0x8664 # 34404
                        ARM_VAL = 0x1c0   # 448
                        ARM64_VAL = 0xaa64 # 43620

                        if machine_val == AMD64_VAL:
                            logger.debug("Setting Capstone mode to 64-bit")
                            pe_mode = capstone.CS_MODE_64
                        elif machine_val == ARM_VAL:
                            logger.debug("Setting Capstone arch/mode to ARM")
                            pe_arch = capstone.CS_ARCH_ARM
                            pe_mode = capstone.CS_MODE_ARM
                        elif machine_val == ARM64_VAL:
                            logger.debug("Setting Capstone arch/mode to ARM64")
                            pe_arch = capstone.CS_ARCH_ARM64
                            pe_mode = capstone.CS_MODE_ARM
                        elif machine_val == I386_VAL:
                            logger.debug("Machine is I386, using default Capstone X86/32-bit")
                            pass # Defaults are already X86/32-bit
                        else:
                            logger.warning(f"Unhandled PE machine type value: {machine_val}. Using default X86/32-bit.")
                    else:
                        logger.warning("Machine type object does not have '.value'. Cannot determine arch/mode accurately. Using default.")
                    # --- End machine type value comparison ---

                # Call opcode extractor
                opcode_entropy, opcode_histogram = extract_opcode_features_py(
                    entrypoint_bytes, TOP_N_OPCODES, arch=pe_arch, mode=pe_mode
                )
                # Assign features
                all_features['exec_entrypoint_mean'] = byte_mean
                all_features['exec_entrypoint_opcode_entropy'] = opcode_entropy
                for i, val in enumerate(opcode_histogram):
                    all_features[f'exec_entrypoint_opcode_hist_{i}'] = float(val)

            except Exception as opcode_err:
                logger.error(f"Error during PE entrypoint/opcode extraction for {path}: {opcode_err}", exc_info=True)

        elif is_elf and binary:
            logger.debug(f"Calling extract_features_elf for {path}")
            # Assumes extract_features_elf is defined
            elf_feats = extract_features_elf(path, raw_bytes, binary)
            all_features.update(elf_feats)
            # TODO: Add ELF specific generics if needed

        elif is_macho and binary:
            logger.debug(f"Calling extract_features_macho for {path}")
            # Assumes extract_features_macho is defined
            macho_feats = extract_macho_features(path, raw_bytes, binary)
            all_features.update(macho_feats)
            # TODO: Add Mach-O specific generics if needed

        elif is_pdf:
            # Assumes PDF_LIB_AVAILABLE is defined globally
            if PDF_LIB_AVAILABLE:
                logger.debug(f"Calling extract_features_pdf for {path}")
                # Assumes extract_features_pdf is defined
                pdf_feats = extract_features_pdf(path, raw_bytes)
                all_features.update(pdf_feats)
        elif is_script:
            logger.debug(f"Calling extract_features_script for {path}")
            # Assumes extract_features_script is defined
            script_feats = extract_features_script(path, raw_bytes)
            all_features.update(script_feats)
        elif not (is_pe or is_elf or is_macho or is_pdf or is_script): # Log if no type was identified
            logger.info(f"File {path} not identified as any specific type. Using general features only.")

        # --- 5. Assemble Vector based on FINAL_FEATURE_ORDER ---
        # Create the vector based on the defined order, getting values from all_features dict
        output_vector_ordered = [0.0] * len(FINAL_FEATURE_ORDER)
        for i, feature_name in enumerate(FINAL_FEATURE_ORDER):
            output_vector_ordered[i] = float(all_features.get(feature_name, 0.0))

        # --- 6. Pad or Truncate to TARGET_VECTOR_SIZE ---
        current_len = len(output_vector_ordered)
        if current_len < TARGET_VECTOR_SIZE:
            padding_needed = TARGET_VECTOR_SIZE - current_len
            output_vector = output_vector_ordered + ([0.0] * padding_needed)
            logger.debug(f"Padded feature vector with {padding_needed} zeros.")
        elif current_len > TARGET_VECTOR_SIZE:
            logger.warning(f"Defined features ({current_len}) exceed target size ({TARGET_VECTOR_SIZE}). Truncating vector!")
            output_vector = output_vector_ordered[:TARGET_VECTOR_SIZE]
        else:
            output_vector = output_vector_ordered # Sizes match

        logger.debug(f"Feature extraction complete for {path}. Final vector size: {len(output_vector)}")

        logger.debug("\n" + "="*25 + f" Features for {os.path.basename(path)} " + "="*25)
        max_name_len = 0
        if FINAL_FEATURE_ORDER:
            # Calculate padding width based on longest feature name
            max_name_len = max(len(name) for name in FINAL_FEATURE_ORDER)

        # Iterate through the DEFINED feature order and print name/value from the calculated vector
        for i, name in enumerate(FINAL_FEATURE_ORDER):
            # Get value from the ordered vector (before padding/truncation)
            # Ensure index is within bounds of the ordered vector
            value = output_vector_ordered[i] if i < len(output_vector_ordered) else 0.0
            # Format: [Index] Name (padded) : Value
            # Adjust index padding based on total number of defined features
            idx_padding = len(str(len(FINAL_FEATURE_ORDER)))
            logger.debug(f"[{i:{idx_padding}}] {name:<{max_name_len}} : {value:>.8f}") # 8 decimal places

        # Print summary of padding/truncation/final size
        if current_len < TARGET_VECTOR_SIZE:
            logger.debug(f"\n... Vector padded with {TARGET_VECTOR_SIZE - current_len} zeros to reach target size {TARGET_VECTOR_SIZE}.")
        elif current_len > TARGET_VECTOR_SIZE:
            logger.debug(f"\n... Vector truncated from {current_len} features to target size {TARGET_VECTOR_SIZE}.")
        logger.debug(f"Total named features listed: {len(FINAL_FEATURE_ORDER)}")
        logger.debug(f"Final vector size returned: {len(output_vector)}")
        logger.debug("="*(52 + len(os.path.basename(path))) + "\n")

    # --- Outer Exception Handler ---
    except Exception as e:
        logger.error(f"Unhandled error during feature extraction orchestrator for {path}: {e}", exc_info=True)
        # Return default zero vector on major failure, ensure size matches target
        output_vector = [0.0] * TARGET_VECTOR_SIZE

    finally:
        # Explicitly delete LIEF object if it exists
        if binary is not None:
            logger.debug(f"Deleting LIEF binary object for {path}")
            try:
                del binary
            except NameError: # Should not happen, but defensive
                pass

    # Ensure the returned vector has the correct target size, even after errors
    if len(output_vector) != TARGET_VECTOR_SIZE:
        logger.error(f"Final vector size mismatch! Expected {TARGET_VECTOR_SIZE}, got {len(output_vector)}. Returning zero vector.")
        return [0.0] * TARGET_VECTOR_SIZE

    return output_vector

# --- Main Execution Logic ---
def process_file(path):
    """Function to process a single file and print features."""
    logger.info(f"Processing file: {path}")
    start_proc_time = time.time()
    try:
        feats = extract_features(path)
        # Output comma-separated floats
        print(",".join(map(lambda x: f"{x:.6f}", feats))) # Format floats
        logger.info(f"Successfully processed {path} in {time.time() - start_proc_time:.2f}s")
    except Exception as e:
         logger.critical(f"CRITICAL FAILURE processing {path}: {e}", exc_info=True)
         # Optionally print an error indicator or an empty vector line
         print(",".join(['0.0'] * FINAL_VECTOR_SIZE)) # Print default vector on critical failure

def process_directory(directory_path, max_files=None): # Changed num_files_to_process to max_files
    """Function to process files in a directory."""
    files_processed = 0
    logger.info(f"Processing directory: {directory_path}" + (f" (max_files={max_files})" if max_files else ""))

    try:
        for entry in os.scandir(directory_path):
            if entry.is_file():
                process_file(entry.path)
                files_processed += 1
                if max_files is not None and files_processed >= max_files:
                    logger.info(f"Reached max files limit ({max_files}). Stopping directory scan.")
                    break
            elif entry.is_dir():
                 logger.debug(f"Skipping subdirectory: {entry.path}") # Optionally recurse later
    except FileNotFoundError:
         logger.error(f"Directory not found: {directory_path}")
    except OSError as e:
         logger.error(f"Error scanning directory {directory_path}: {e}")

    logger.info(f"Finished processing directory {directory_path}. Processed {files_processed} files.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <file_or_directory_path> [max_files]")
        sys.exit(1)

    input_path = sys.argv[1]
    max_files_arg = None
    if len(sys.argv) > 2:
        try:
            max_files_arg = int(sys.argv[2])
            if max_files_arg <= 0: raise ValueError()
        except ValueError:
            print("Error: [max_files] argument must be a positive integer.")
            sys.exit(1)


    if os.path.isdir(input_path):
        process_directory(input_path, max_files=max_files_arg)
    elif os.path.isfile(input_path):
        process_file(input_path)
    else:
        print(f"❌ Invalid path: {input_path}. It must be a valid file or directory.")
        sys.exit(1)

    logger.info("Script finished.")
