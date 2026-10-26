"""Record the machine and software a run executed on.

Results are only meaningful together with the environment that produced them
(speed depends on the GPU, tiny numeric differences depend on library
versions...). Every run therefore writes ``environment.json``.

Library versions are read from package metadata, so recording the environment
does not import heavy libraries such as PyTorch.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_FIELDS = (
    "timestamp_utc",
    "os",
    "python_version",
    "python_implementation",
    "machine",
    "cpu",
    "cpu_count",
    "memory_total_gb",
    "gpus",
    "packages",
)

TRACKED_PACKAGES = ("numpy", "opencv-python", "PyYAML", "pytest", "torch", "torchvision")


class EnvironmentRecordError(ValueError):
    """Raised when an environment record is incomplete."""


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None  # explicitly "not installed"
    return versions


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _total_memory_gb() -> Optional[float]:
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / 1024**3, 2)
            return None
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 2)
    except Exception:  # noqa: BLE001 - recording must never crash a run
        return None
    return None


def _gpu_info() -> Dict[str, Any]:
    """Query NVIDIA GPUs through ``nvidia-smi``; report *why* if unavailable."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except FileNotFoundError:
        return {"available": False, "devices": [], "reason": "nvidia-smi not found"}
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "devices": [], "reason": f"nvidia-smi failed: {exc}"}
    if result.returncode != 0:
        return {
            "available": False,
            "devices": [],
            "reason": f"nvidia-smi exited with code {result.returncode}",
        }
    devices: List[Dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        name, _, memory = line.rpartition(",")
        try:
            devices.append({"name": name.strip(), "memory_total_mib": int(memory.strip())})
        except ValueError:
            continue
    if not devices:
        return {"available": False, "devices": [], "reason": "nvidia-smi reported no devices"}
    return {"available": True, "devices": devices, "reason": None}


def collect_environment() -> Dict[str, Any]:
    """Gather the environment record as a plain dict."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": _total_memory_gb(),
        "gpus": _gpu_info(),
        "packages": _package_versions(),
    }


def validate_environment_record(record: Dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise EnvironmentRecordError(f"environment record is missing field(s): {missing}")


def write_environment_record(path: Path | str) -> Dict[str, Any]:
    """Collect, validate and write the environment record; returns the record."""
    record = collect_environment()
    validate_environment_record(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record
