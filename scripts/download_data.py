#!/usr/bin/env python3
"""Download Gravity Spy dataset from Zenodo.

This script downloads the Gravity Spy training set from Zenodo
(DOI: 10.5281/zenodo.1476156) which contains spectrogram images of
glitches from LIGO gravitational wave detector data.

Dataset files:
- trainingset_v1d0_metadata.csv: Labels and sample types
- trainingsetv1d0.h5: HDF5 file with spectrogram arrays (3.3 GB)
- trainingsetv1d0.tar.gz: Raw PNG images (9.6 GB, optional)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --include-png  # Also download PNG archive
"""

import argparse
import hashlib
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# Zenodo record ID
ZENODO_RECORD_ID = "1476156"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "gravityspy"

# Files to download (filename: expected MD5 checksum if known)
REQUIRED_FILES = [
    "trainingset_v1d0_metadata.csv",
    "trainingsetv1d0.h5",
]

OPTIONAL_FILES = [
    "trainingsetv1d0.tar.gz",  # Large PNG archive
]


def get_zenodo_files():
    """Fetch file information from Zenodo API.

    Returns:
        dict: Mapping of filename to file info (url, size, checksum).
    """
    print(f"Fetching file info from Zenodo record {ZENODO_RECORD_ID}...")
    response = requests.get(ZENODO_API_URL)
    response.raise_for_status()
    record = response.json()

    files = {}
    for f in record.get("files", []):
        files[f["key"]] = {
            "url": f["links"]["self"],
            "size": f["size"],
            "checksum": f.get("checksum", ""),
        }

    return files


def download_file(url: str, dest_path: Path, expected_size: int = None):
    """Download a file with progress bar.

    Args:
        url: URL to download from.
        dest_path: Path to save the file.
        expected_size: Expected file size in bytes (for progress bar).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = expected_size or int(response.headers.get("content-length", 0))

    with (
        open(dest_path, "wb") as f,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify MD5 checksum of a file.

    Args:
        file_path: Path to the file.
        expected_checksum: Expected MD5 checksum (format: "md5:hash").

    Returns:
        bool: True if checksum matches or no checksum provided.
    """
    if not expected_checksum or not expected_checksum.startswith("md5:"):
        return True

    expected_hash = expected_checksum.replace("md5:", "")
    print(f"Verifying checksum for {file_path.name}...")

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    actual_hash = md5.hexdigest()
    if actual_hash != expected_hash:
        print(f"  Checksum mismatch! Expected {expected_hash}, got {actual_hash}")
        return False

    print("  Checksum verified.")
    return True


def extract_tar(tar_path: Path, dest_dir: Path):
    """Extract tar.gz archive with progress.

    Args:
        tar_path: Path to the tar.gz file.
        dest_dir: Directory to extract to.
    """
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, dest_dir)
    print(f"Extracted to {dest_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Gravity Spy dataset from Zenodo"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to save data (default: data/gravityspy)",
    )
    parser.add_argument(
        "--include-png",
        action="store_true",
        help="Also download the large PNG archive (9.6 GB)",
    )
    parser.add_argument(
        "--extract-png",
        action="store_true",
        help="Extract PNG archive after download (requires --include-png)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip checksum verification",
    )
    args = parser.parse_args()

    # Get file info from Zenodo
    zenodo_files = get_zenodo_files()

    # Determine which files to download
    files_to_download = REQUIRED_FILES.copy()
    if args.include_png:
        files_to_download.extend(OPTIONAL_FILES)

    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading to: {args.data_dir}")

    # Download each file
    for filename in files_to_download:
        if filename not in zenodo_files:
            print(f"\nWarning: {filename} not found in Zenodo record")
            continue

        file_info = zenodo_files[filename]
        dest_path = args.data_dir / filename

        # Skip if already exists
        if dest_path.exists():
            if dest_path.stat().st_size == file_info["size"]:
                print(f"\n{filename} already exists, skipping download")
                continue
            else:
                print(f"\n{filename} exists but size differs, re-downloading")

        print(f"\nDownloading {filename} ({file_info['size'] / 1e9:.1f} GB)...")
        download_file(file_info["url"], dest_path, file_info["size"])

        # Verify checksum
        if not args.skip_verify:
            if not verify_checksum(dest_path, file_info["checksum"]):
                print(f"Warning: Checksum verification failed for {filename}")

    # Extract PNG archive if requested
    if args.extract_png and args.include_png:
        tar_path = args.data_dir / "trainingsetv1d0.tar.gz"
        if tar_path.exists():
            extract_tar(tar_path, args.data_dir)

    print("\nDownload complete!")
    print(f"\nDataset saved to: {args.data_dir}")
    print("\nFiles:")
    for f in args.data_dir.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
