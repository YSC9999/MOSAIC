"""Extract the RWF-2000 Kaggle zip with Windows-safe file names (resumable, CRC-checked).

Names that are non-ASCII or longer than 80 characters (the archive contains garbled, very long names that
exceed MAX_PATH) are replaced by ``x<sha1(base)[:12]>_<n>.avi``; the trailing clip number is kept so
source-video grouping still works. A JSON map new_name -> original zip name is written next to the data.
Usage: python scripts/extract_rwf_zip.py ZIP DEST_DIR [--delete-zip-when-verified]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

SAFE = re.compile(r"^[A-Za-z0-9._\- ]{1,80}$")


def safe_name(orig_path: str) -> str:
    parent, name = orig_path.rsplit("/", 1)
    if SAFE.match(name):
        return f"{parent}/{name}"
    stem = name.rsplit(".", 1)[0]
    m = re.search(r"_(\d+)$", stem)
    base, n = (stem[: m.start()], m.group(1)) if m else (stem, "0")
    return f"{parent}/x{hashlib.sha1(base.encode('utf-8', 'surrogatepass')).hexdigest()[:12]}_{n}.avi"


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    delete = "--delete-zip-when-verified" in sys.argv
    z = zipfile.ZipFile(src)
    infos = [i for i in z.infolist() if i.filename.endswith(".avi")]
    mapping, used = {}, {}
    for i in infos:
        assert not i.filename.startswith("/") and ".." not in i.filename.split("/"), i.filename
        new = safe_name(i.filename)
        if new in used:
            raise RuntimeError(f"name collision: {new} <- {i.filename} and {used[new]}")
        used[new] = i.filename
        mapping[new] = i.filename
    done = skipped = 0
    for i in infos:
        new = safe_name(i.filename)
        out = dst / new
        if out.exists() and out.stat().st_size == i.file_size:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        with z.open(i) as fi, tmp.open("wb") as fo:        # reading to the end verifies the CRC
            while chunk := fi.read(1 << 20):
                fo.write(chunk)
        tmp.replace(out)
        done += 1
        if done % 250 == 0:
            print(f"extracted {done} (skipped {skipped})", flush=True)
    (dst / "name_map.json").write_text(json.dumps(mapping, ensure_ascii=True, indent=0), encoding="utf-8")
    bad = [n for n, o in mapping.items() if not (dst / n).exists() or (dst / n).stat().st_size != z.getinfo(o).file_size]
    files = sum(1 for _, _, fs in os.walk(dst) for f in fs if f.endswith(".avi"))
    renamed = sum(1 for n, o in mapping.items() if n != o)
    print(f"finished: extracted={done} skipped={skipped} files_on_disk={files} expected={len(infos)} renamed={renamed} bad={len(bad)}", flush=True)
    ok = files == len(infos) and not bad
    if ok and delete:
        z.close()
        os.remove(src)
        print("verified; deleted zip", flush=True)
    elif not ok:
        print("NOT verified; zip kept", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
