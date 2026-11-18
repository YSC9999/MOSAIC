"""Fast parallel downloader for RWF-2000 from Hugging Face A1mal/RWF-2000-Dataset.

Downloads all 2,000 real surveillance clips (1.22 GB total) into:
    data/RWF-2000/
      train/
        Fight/
        NonFight/
      val/
        Fight/
        NonFight/
"""

from __future__ import annotations

import os
import sys
import time
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/A1mal/RWF-2000-Dataset/resolve/main"

SPLIT_MAP = {
    "data/RWF-2000 Sliced/train/Fight": ("train", "Fight"),
    "data/RWF-2000 Sliced/train/NonFight": ("train", "NonFight"),
    "data/RWF-2000 Sliced/val/Fight": ("val", "Fight"),
    "data/RWF-2000 Sliced/val/NonFight": ("val", "NonFight"),
}


def fetch_file_list() -> list[tuple[str, Path, int]]:
    tasks = []
    print("Fetching file list from Hugging Face API...")
    for hf_split, (local_split, local_class) in SPLIT_MAP.items():
        url = f"https://huggingface.co/api/datasets/A1mal/RWF-2000-Dataset/tree/main/{urllib.parse.quote(hf_split)}"
        req = urllib.request.Request(url, headers={"User-Agent": "mosaic-fl/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        for item in items:
            if item.get("type") == "file":
                rel_path = item["path"]
                fname = Path(rel_path).name
                size = item.get("size", 0)
                download_url = f"{HF_BASE}/{urllib.parse.quote(rel_path)}"
                dest = Path("data/RWF-2000") / local_split / local_class / fname
                tasks.append((download_url, dest, size))
    print(f"Total files to download: {len(tasks)}")
    return tasks


def download_one(url: str, dest: Path, expected_size: int) -> bool:
    if dest.is_file() and dest.stat().st_size == expected_size and expected_size > 0:
        return True  # already downloaded
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mosaic-fl/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_dest, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if temp_dest.is_file():
            temp_dest.replace(dest)
            return True
    except Exception as e:
        if temp_dest.is_file():
            temp_dest.unlink(missing_ok=True)
        return False
    return False


def main():
    start = time.time()
    tasks = fetch_file_list()
    total = len(tasks)
    total_bytes = sum(t[2] for t in tasks)
    print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB")

    completed = 0
    failed = 0
    num_workers = 12

    print(f"Starting parallel download with {num_workers} threads...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_one, url, dest, size): (dest, size) for url, dest, size in tasks}
        for future in as_completed(futures):
            dest, size = futures[future]
            try:
                ok = future.result()
                if ok:
                    completed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - start
                pct = (completed / total) * 100
                print(f"[{completed}/{total}] ({pct:.1f}%) downloaded in {elapsed:.1f}s...")

    elapsed = time.time() - start
    print(f"Download complete: {completed} files downloaded, {failed} failed in {elapsed:.1f}s.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
