# StaticAnalysisFeaturesPy

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A standalone Python tool for extracting static features from various file types (PE, ELF, MachO, PDF, Scripts), primarily focused on applications in security analysis, malware research, and generating datasets for Machine Learning models.

This tool leverages powerful libraries like LIEF, Yara, and Capstone to perform deep static analysis without executing the input files.

## 📚 Table of Contents
- [Core Features](#-core-features)
- [Motivation & Use Cases](#-motivation--use-cases)
- [File Structure](#-file-structure)
- [Setup & Installation](#️-setup--installation)
- [Configuration](#-configuration)
- [Usage](#️-usage)
- [Feature Vector Output](#-feature-vector-output)
- [License](#️-license)

## ✨ Core Features

* **Multi-Format Support:** Extracts features from PE (Windows), ELF (Linux), and Mach-O (macOS) executables, as well as PDFs and generic script files.
* **Deep Static Analysis:** Uses specialized libraries for detailed parsing:
    * **LIEF Engine:** Parses executable formats (PE, ELF, MachO) for headers, sections, imports, exports, resources, signatures, etc.
    * **Yara-Python:** Applies custom YARA rules for pattern matching.
    * **Capstone Engine:** Disassembles entrypoint code for opcode analysis (histograms, entropy).
    * **python-magic:** Identifies file types based on magic numbers.
    * **PyPDF2:** Extracts metadata and structural info from PDF files (optional).
* **Rich Feature Set:** Generates a configurable, fixed-length feature vector including:
    * General file characteristics (size, entropy, type flags).
    * String features (counts of IPs/URLs, suspicious string hits, lengths, Base64-like detection).
    * YARA rule match indicators.
    * Format-specific details (PE sections/imports/exports/resources/Rich Header, ELF segments/sections/symbols, Mach-O commands/sections, PDF metadata/structure, Script characteristics).
* **Configurable & Extensible:** Feature set defined in 'feature_extractor.py' ('FINAL_FEATURE_ORDER') can be customized. Relies on external 'yara_rules/' and 'suspicious_strings.txt'.
* **Batch Processing:** Includes 'generate_features.py' script for efficient parallel processing of large numbers of files from directories or CSV manifests.
* **Flexible Output:** 'generate_features.py' can output extracted features as JSON Lines ('.jsonl') or Apache Parquet ('.parquet').

## 💡 Motivation & Use Cases

* **ML Dataset Generation:** Create datasets for training malware classifiers or other security-focused models.
* **Automated Triage:** Integrate into analysis pipelines to quickly gather static properties of submitted files.
* **Security Research:** Facilitate large-scale static analysis of binary datasets.

## 📁 File Structure

```
StaticAnalysisFeaturesPy/
│
├── feature_extractor.py      # Core extraction logic, feature definitions
├── generate_features.py      # Batch processing runner script
│
├── yara_rules/               # Directory for YARA rule files
│   └── *.yar                 # Example rule file(s)
│
├── training/                 # Helper scripts for training data prep
│   └── prepare_csv_for_training.py # Script to generate CSV from sample dirs
│
├── suspicious_strings.txt    # List of suspicious strings (one per line)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # Apache 2.0 License text
```

## ⚙️ Setup & Installation

**Prerequisites:**

* **Python:** 3.9+ recommended. **Note:** While the code may work on newer versions like 3.12, using a slightly older common version like **Python 3.10 or 3.11** might offer broader compatibility for installing binary dependencies like LIEF and Capstone, especially if pre-built wheels are not available for the absolute latest Python release on your specific OS/architecture.
* **System Libraries:**
    * 'libmagic': Required by 'python-magic'. See installation instructions below. (Often handled by 'python-magic-bin' on Windows).
* **Git:** For cloning the repository.
* **(Optional) C++ Build Tools:** May be needed if installing dependencies like 'lief' fails to find a pre-built wheel for your platform (less common now).

**Installation Steps:**

1. **Clone Repository:**
```bash
    # Replace with your actual repository URL
    git clone https://github.com/your_username/StaticAnalysisFeaturesPy.git
    cd StaticAnalysisFeaturesPy
```

2. **Create & Activate Virtual Environment (Recommended):**
```bash
    python -m venv .venv
    # Linux/macOS
    source .venv/bin/activate
    # Windows CMD/PowerShell
    .\.venv\Scripts\activate
```

3.  **Install System Dependencies:**
* **'libmagic'** (Linux/macOS):
```bash
        # Debian/Ubuntu
        sudo apt-get update && sudo apt-get install -y libmagic1
        # Fedora/RHEL
        sudo dnf install file-libs
        # macOS (Homebrew)
        brew install libmagic
```
* **Windows**: Typically handled by 'python-magic-bin' installed via pip

4.  **Install Python Dependencies:**
```bash
    pip install -r requirements.txt
```
    *(Note: This will install optional dependencies like 'pyarrow' and 'tqdm' too. You can remove them from 'requirements.txt' if not needed.)*

## 🔧 Configuration

* **YARA Rules:** Place your '.yar' or '.yara' files inside the 'yara_rules/' directory. 'feature_extractor.py' will compile them at startup.
* **Suspicious Strings:** Add any suspicious strings (one per line) you want to search for in 'suspicious_strings.txt'.
* **Feature Vector:**
    * The order and content of the final feature vector are defined by the 'FINAL_FEATURE_ORDER' list in 'feature_extractor.py'.
    * Constants like 'TARGET_VECTOR_SIZE' (currently 256), 'MAX_SECTION_RATIOS', 'MAX_ENTROPY_BUCKETS', 'TOP_N_OPCODES' can also be adjusted in 'feature_extractor.py'.

## ▶️ Usage

This tool provides two main ways to extract features:

### 1. Batch Processing ('generate_features.py')

This is the recommended method for processing multiple files and generating datasets. It uses multiprocessing to parallelize feature extraction.

**Arguments:**

* '--benign <dir>' & '--malware <dir>': Specify directories containing benign and malware samples. The script will assign labels 0 and 1 respectively.
* '--csv <file>' & '--samples <dir>': Specify a CSV file (with 'id', 'sha256', 'list' columns) and the path where the corresponding sample files are located (named either by id or sha256).
* '--jsonl <outfile>': Path to output results as JSON Lines (one JSON object per file, per line).
* '--parquet <outfile>': (Requires 'pyarrow') Path to write results to an Apache Parquet file.
* '--workers <N>': Number of parallel worker processes to use (default: CPU count).
* '--dump-names': Don't process files, just generate the 'names.txt' file (see below).
* '--output-names-file <name>': Filename to use when dumping feature names (default: 'names.txt').

**Examples:**

```bash
# Activate your Python environment first!

# Process benign/malware directories, outputting Parquet file using 8 workers
python generate_features.py \
    --benign /path/to/benign/samples \
    --malware /path/to/malware/samples \
    --parquet dataset.parquet \
    --workers 8

# Process samples listed in CSV from a samples directory, outputting JSON Lines file
python generate_features.py \
    --csv my_samples.csv \
    --samples /path/to/all/samples \
    --jsonl dataset.jsonl
```

### 2. Single File / Directory Extraction ('feature_extractor.py')

You can also run 'feature_extractor.py' directly on a single file or a directory. It will print a fixed-length, comma-separated feature vector to standard output for each file processed. This is useful for quick checks or integration with tools expecting CSV output.

**Example:**

```bash
# Activate your Python environment first!
python feature_extractor.py /path/to/your/sample.exe
```

Example Output (truncated):

```
118784.000000,7.051762,0.265785,0.106479,...,0.000000
```

### 3. Dumping Feature Names

To get an ordered list of all feature names (including any padding placeholders, matching the 'TARGET_VECTOR_SIZE'), run 'generate_features.py' with '--dump-names':

```bash
# Activate your Python environment first!

# Default output: names.txt
python generate_features.py --dump-names

# Custom filename
python generate_features.py --dump-names --output-names-file custom_names.txt
```
* Creates 'names.txt' (or your custom name) with one feature name per line. This list matches the order of values in the feature vector.

### 4. Preparing Training CSV (`training/prepare_csv_for_training.py`)

This script helps generate the necessary CSV manifest file used as input for batch feature generation (`generate_features.py --csv ...`). It scans directories of samples, calculates basic metadata (hashes, size, entropy, type), and assigns labels. This is useful for creating the input CSV required by other parts of the workflow if you only have directories of labeled samples.

**Arguments:**

* `--suffix <suffix>`: (Required) A suffix to append to generated numeric IDs to ensure uniqueness across different runs/datasets (e.g., 'B' for benign, 'M' for malware).
* `--path <dir>`: (Required) Path to the directory containing the sample files to scan.
* `--label <label>`: (Required) The label to assign ('Whitelist' or 'Blacklist').
* `--output <outfile.csv>`: (Required) Path to the output CSV file.
* `--workers <N>`: Number of parallel worker processes to use (default: CPU count).

**Example:**

```bash
# Activate your Python environment first!
# Generate CSV metadata for benign samples
python training/prepare_csv_for_training.py \
    --suffix B1 --path /path/to/benign/samples --label Whitelist \
    --output benign_list.csv --workers 8

# Generate CSV metadata for malware samples
python training/prepare_csv_for_training.py \
    --suffix M1 --path /path/to/malware/samples --label Blacklist \
    --output malware_list.csv --workers 8
```

* Creates '.csv' files with calculated metadata. You would typically combine outputs from multiple runs (e.g., benign and malware) into a single CSV for the next step.

## 📊 Feature Vector Output

The primary output from batch processing ('generate_features.py') is either a '.jsonl' or '.parquet' file containing columns/fields like:

* 'name': Identifier (filename or CSV id/sha256).
* 'label': Integer label (0 or 1).
* 'sha256': Calculated SHA256 hash of the file.
* 'features': The list or array of floating-point feature values (always length 'TARGET_VECTOR_SIZE', e.g., 256).

The exact meaning and order of the values in the 'features' list are defined by 'FINAL_FEATURE_ORDER' in 'feature_extractor.py' and can be obtained by running 'python generate_features.py --dump-names'.

## 🛡️ License

This project is licensed under the Apache License, Version 2.0 - see the 'LICENSE' file for details.
