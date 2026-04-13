"""

CLI entry-point: download the Kaggle anime dataset into data/raw/.
"""


import os
import shutil
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds


def check_kaggle_auth() -> bool:
    username    = os.getenv("KAGGLE_USERNAME")
    key         = os.getenv("KAGGLE_KEY")
    json_exists = Path("~/.kaggle/kaggle.json").expanduser().exists()

    if json_exists or (username and key):
        return True

    print("ERROR: Kaggle credentials not found.")
    print("  Set either:")
    print("    1) ~/.kaggle/kaggle.json")
    print("    2) env vars KAGGLE_USERNAME and KAGGLE_KEY")
    return False


def download_dataset(dataset_slug: str, target_path: str) -> str:
    """Download *dataset_slug* from Kaggle into *target_path*.

    Uses kagglehub's built-in resume support; retries up to MAX_RETRIES times
    on network timeouts.  Returns the final destination path.
    """
    import kagglehub  # lazy — only needed when actually downloading

    target_path = Path(target_path).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"[download] Fetching dataset  : {dataset_slug}")
    print(f"[download] Destination       : {target_path}")

    cached_path = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            cached_path = kagglehub.dataset_download(dataset_slug)
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            TimeoutError,
        ) as exc:
            if attempt == MAX_RETRIES:
                print(f"[download] All {MAX_RETRIES} attempts failed. Giving up.")
                raise
            print(f"[download] Timeout on attempt {attempt}/{MAX_RETRIES}: {exc}")
            print(f"[download] Retrying in {RETRY_DELAY}s … (kagglehub will resume)")
            time.sleep(RETRY_DELAY)

    print(f"[download] Cached at         : {cached_path}")

    cached_path = Path(cached_path)
    if cached_path.is_dir():
        for item in cached_path.iterdir():
            dst = target_path / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
    else:
        shutil.copy2(cached_path, target_path)

    print(f"[download] Stored at         : {target_path}")
    return str(target_path)


def main() -> None:
    """Download the Kaggle anime dataset into data/raw/."""
    if not check_kaggle_auth():
        sys.exit(1)

    dataset = os.getenv("KAGGLE_DATASET", "diraizel/anime-images-dataset")
    raw_dir = os.getenv("DATA_RAW_DIR", str(PROJECT_ROOT / "data" / "raw"))
    download_dataset(dataset, raw_dir)


if __name__ == "__main__":
    main()