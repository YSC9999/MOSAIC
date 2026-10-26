"""Resumable downloader and extractor for the RWF-2000 dataset.

Downloads the RWF-2000 archive from Hugging Face Datasets
(``DanJoshua/RWF-2000``) with HTTP Range resume support and extracts it into
the standard directory structure expected by ``mosaic``:

    <target_dir>/
      train/
        Fight/
        NonFight/
      val/
        Fight/
        NonFight/

Usage:
    python scripts/download_rwf2000.py --out data/RWF-2000
    python -m mosaic download-rwf --out data/RWF-2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

HF_REPO = "DanJoshua/RWF-2000"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
DEFAULT_ARCHIVE_NAME = "RWF-2000.tar.gz"


def list_hf_files() -> List[str]:
    """Query Hugging Face API for files in the dataset repository."""
    req = urllib.request.Request(HF_API_URL, headers={"User-Agent": "mosaic-fl-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.loads(resp.read().decode("utf-8"))
    return [item["path"] for item in items if isinstance(item, dict) and "path" in item]


def download_with_resume(url: str, dest: Path, chunk_size: int = 4 * 1024 * 1024) -> Path:
    """Download ``url`` to ``dest`` with automatic HTTP Range resume and progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = dest.stat().st_size if dest.is_file() else 0

    headers = {"User-Agent": "mosaic-fl-downloader/1.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # Range Not Satisfiable -> already fully downloaded
            print(f"Archive already completely downloaded: {dest} ({existing_bytes / 1024**3:.2f} GB)")
            return dest
        raise

    with resp:
        status = getattr(resp, "status", 200)
        mode = "ab" if (status == 206 and existing_bytes > 0) else "wb"
        if mode == "wb":
            existing_bytes = 0

        content_len = resp.headers.get("Content-Length")
        total_bytes = (existing_bytes + int(content_len)) if content_len is not None else None

        downloaded = existing_bytes
        last_pct = -1
        with dest.open(mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes and total_bytes > 0:
                    pct = int(downloaded * 100 // total_bytes)
                    if pct != last_pct and (pct % 2 == 0 or pct == 100):
                        last_pct = pct
                        gb_done = downloaded / 1024**3
                        gb_total = total_bytes / 1024**3
                        print(
                            f"Downloading {dest.name}: {pct:3d}% ({gb_done:.2f} / {gb_total:.2f} GB)",
                            flush=True,
                        )
                else:
                    gb_done = downloaded / 1024**3
                    print(f"\rDownloading {dest.name}: {gb_done:.2f} GB", end="", flush=True)
        print()
    return dest


def _sanitize_filename(name: str) -> str:
    """Ensure a filename is Windows- and OpenCV-safe pure ASCII without collisions."""
    stem, ext = os.path.splitext(name)
    # Check if already clean ASCII
    if re.fullmatch(r"[A-Za-z0-9_.\- ()]+", stem) and len(name) <= 120:
        return name
    digest = hashlib.sha1(name.encode("utf-8", errors="surrogatepass")).hexdigest()[:10]
    ascii_tail = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    if len(ascii_tail) > 40:
        ascii_tail = ascii_tail[-40:]
    clean_stem = f"rwf_{digest}_{ascii_tail}" if ascii_tail else f"rwf_{digest}"
    return f"{clean_stem}{ext}"


def _normalize_extracted_layout(target_dir: Path) -> None:
    """If the archive extracted into a nested ``RWF-2000/`` subfolder, hoist ``train/`` and ``val/``."""
    nested = target_dir / "RWF-2000"
    if nested.is_dir():
        for split_name in ("train", "val"):
            src_split = nested / split_name
            if src_split.is_dir():
                for cls_dir in src_split.iterdir():
                    if cls_dir.is_dir():
                        dst_cls = target_dir / split_name / cls_dir.name
                        dst_cls.mkdir(parents=True, exist_ok=True)
                        for f in cls_dir.iterdir():
                            if f.is_file():
                                clean_name = _sanitize_filename(f.name)
                                dst_file = dst_cls / clean_name
                                if not dst_file.exists():
                                    shutil.move(str(f), str(dst_file))
                                else:
                                    f.unlink()
        shutil.rmtree(nested, ignore_errors=True)


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract ``.tar.gz``, ``.tar``, or ``.zip`` archive into ``target_dir``."""
    print(f"Extracting {archive_path} into {target_dir} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    _normalize_extracted_layout(target_dir)
    name_lower = archive_path.name.lower()
    if name_lower.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            extracted_count = 0
            skipped_count = 0
            for member in tf:
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if parts and parts[0].lower() == "rwf-2000":
                    parts = parts[1:]
                if not parts:
                    continue
                clean_parts = list(parts[:-1]) + [_sanitize_filename(parts[-1])]
                dest_file = target_dir.joinpath(*clean_parts)
                if dest_file.is_file() and dest_file.stat().st_size == member.size:
                    skipped_count += 1
                    continue
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                src_f = tf.extractfile(member)
                if src_f is not None:
                    with src_f, dest_file.open("wb") as out_f:
                        shutil.copyfileobj(src_f, out_f)
                    extracted_count += 1
                    if (extracted_count + skipped_count) % 250 == 0:
                        print(
                            f"  Progress: {extracted_count + skipped_count} clips processed "
                            f"({extracted_count} newly extracted, {skipped_count} already on disk)...",
                            flush=True,
                        )
            print(
                f"Extraction complete: {extracted_count} newly extracted, "
                f"{skipped_count} already on disk."
            )
    elif name_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=target_dir)
        _normalize_extracted_layout(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")


def download_rwf2000(
    target_dir: Path | str = "data/RWF-2000",
    keep_archive: bool = False,
) -> Path:
    """Download and extract RWF-2000 into ``target_dir``."""
    out = Path(target_dir)
    if (out / "train" / "Fight").is_dir() and (out / "val" / "Fight").is_dir():
        train_count = len(list((out / "train").rglob("*.avi")))
        val_count = len(list((out / "val").rglob("*.avi")))
        if train_count >= 1600 and val_count >= 400:
            print(
                f"RWF-2000 already present at {out.resolve()} "
                f"(train={train_count} clips, val={val_count} clips)."
            )
            return out

    archive_name = DEFAULT_ARCHIVE_NAME
    archive_dest = out.parent / archive_name

    # If the full ~11.47 GB archive is already on disk, extract directly without network calls
    if archive_dest.is_file() and archive_dest.stat().st_size >= 12_000_000_000:
        print(
            f"Found complete archive on disk: {archive_dest.resolve()} "
            f"({archive_dest.stat().st_size / 1024**3:.2f} GB)"
        )
    else:
        try:
            files = list_hf_files()
            archives = [
                f for f in files if f.lower().endswith((".tar.gz", ".tgz", ".zip", ".tar"))
            ]
            if archives:
                archive_name = archives[0]
                archive_dest = out.parent / archive_name
        except Exception as exc:  # noqa: BLE001
            print(f"Note: could not list repo tree ({exc}); using default '{archive_name}'")

        url = f"{HF_RESOLVE_BASE}/{archive_name}"
        print(f"Source URL: {url}")
        print(f"Archive destination: {archive_dest.resolve()}")
        download_with_resume(url, archive_dest)

    extract_archive(archive_dest, out)

    if not keep_archive and archive_dest.is_file():
        archive_dest.unlink()

    train_count = len(list((out / "train").rglob("*.avi")))
    val_count = len(list((out / "val").rglob("*.avi")))
    print(f"RWF-2000 ready at: {out.resolve()} (train={train_count} clips, val={val_count} clips)")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download and extract the RWF-2000 dataset")
    parser.add_argument(
        "--out",
        default="data/RWF-2000",
        help="Target directory for extracted dataset (default: data/RWF-2000)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded .tar.gz archive after extraction",
    )
    args = parser.parse_args(argv)
    try:
        download_rwf2000(target_dir=args.out, keep_archive=args.keep_archive)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"DOWNLOAD ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
