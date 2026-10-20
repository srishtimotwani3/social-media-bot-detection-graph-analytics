"""
00_download_data.py
-------------------
Downloads and extracts the Cresci-2017 dataset into data/raw/.

The dataset is available from the Bot Repository (IU/OSoMe):
  https://botometer.osome.iu.edu/bot-repository/datasets.html

Because the dataset requires a form-based download, this script:
  1. Tries to fetch from a direct Zenodo/GitHub mirror if available.
  2. Otherwise prints clear instructions for manual download.
"""

import os
import sys
import zipfile
import pathlib

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cresci-2017"

# ── Dataset info ──────────────────────────────────────────────────────────────
# The Cresci-2017 dataset is available at IEEE DataPort and the Bot Repository.
# Direct HTTP access requires institutional login at IEEE DataPort.
# The alternative is the Bot Repository page (IU/OSoMe) which offers direct CSVs.

BOT_REPO_URL = "https://botometer.osome.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv.zip"

EXPECTED_SUBDIRS = [
    "genuine_accounts.csv",
    "social_spambots_1.csv",
    "social_spambots_2.csv",
    "social_spambots_3.csv",
    "traditional_spambots_1.csv",
    "fake_followers.csv",
]


def check_already_extracted() -> bool:
    """Return True if at least one expected subdirectory exists."""
    if not RAW_DIR.exists():
        return False
    found = [d for d in EXPECTED_SUBDIRS if (RAW_DIR / d).exists()]
    return len(found) > 0


def try_download() -> bool:
    """Attempt to download the dataset zip from the Bot Repository."""
    try:
        import urllib.request
        zip_path = RAW_DIR.parent / "cresci-2017.csv.zip"
        RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Attempting download from:\n  {BOT_REPO_URL}")
        print("[INFO] This may take a few minutes (~150 MB)...")
        urllib.request.urlretrieve(BOT_REPO_URL, zip_path)
        print(f"[INFO] Downloaded to {zip_path}")
        return True
    except Exception as e:
        print(f"[WARN] Download failed: {e}")
        return False


def extract_zip(zip_path: pathlib.Path):
    """Extract zip into RAW_DIR, and unpack any nested subcategory zips."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path} -> {RAW_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    
    # Check for nested zips in datasets_full.csv
    nested_dir = RAW_DIR / "datasets_full.csv"
    if nested_dir.exists():
        for item in nested_dir.glob("*.csv.zip"):
            print(f"[INFO] Unpacking subcategory: {item.name}")
            with zipfile.ZipFile(item, "r") as sub_zf:
                sub_zf.extractall(RAW_DIR)
    print("[INFO] Extraction complete.")


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              MANUAL DATASET DOWNLOAD REQUIRED                    ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. Go to:                                                        ║
║    https://botometer.osome.iu.edu/bot-repository/datasets.html  ║
║                                                                  ║
║ 2. Find "Cresci-2017" and click Download.                        ║
║                                                                  ║
║ 3. Extract the ZIP so you get this layout:                       ║
║    data/raw/cresci-2017/                                         ║
║       genuine_accounts.csv/                                      ║
║          users.csv                                               ║
║          tweets.csv                                              ║
║       social_spambots_1.csv/   (and _2, _3)                      ║
║       traditional_spambots_1.csv/                                ║
║       fake_followers.csv/                                        ║
║                                                                  ║
║ 4. Then run: python src/01_ingest_clean.py                       ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    if check_already_extracted():
        print(f"[OK] Dataset already present at {RAW_DIR}")
        return

    zip_path = RAW_DIR.parent / "cresci-2017.csv.zip"

    # If zip was manually placed, just extract
    if zip_path.exists():
        extract_zip(zip_path)
        return

    # Try automatic download
    if try_download():
        extract_zip(zip_path)
    else:
        print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
