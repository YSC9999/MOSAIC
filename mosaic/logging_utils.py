"""Logging conventions.

Two kinds of output, kept separate on purpose:

* **Logs** (this module) are for humans reading a console or ``run.log``.
  They go to stderr and, optionally, to a file. Format:
  ``time | LEVEL | logger name | message``.
* **Diagnostics** (``diagnostics.json``) are for programs and for evaluation.
  They are structured data, never parsed out of log text.

``log_event`` writes ``event=<name> key=value ...`` lines, which are easy to
read and easy to search (``findstr event=decode_finished run.log``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_LOGGER_NAME = "mosaic"
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_HANDLER_MARK = "_mosaic_handler"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that lives under the ``mosaic`` namespace."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def shutdown_logging() -> None:
    """Remove and close the handlers installed by ``setup_logging``."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()


def setup_logging(level: str = "INFO", log_file: Optional[Path | str] = None) -> logging.Logger:
    """Configure the ``mosaic`` logger. Safe to call repeatedly (no duplicate handlers)."""
    shutdown_logging()
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(getattr(logging, level))
    root.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_MARK, True)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARK, True)
        root.addHandler(file_handler)
    return root


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log ``event=<name> key=value ...`` with keys in sorted order."""
    parts = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
    logger.info("event=%s %s", event, parts if parts else "")
