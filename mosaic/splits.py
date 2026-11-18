"""Source-video grouped splits for the v3 pipeline (spec section 10).

RWF-2000 cuts several clips from each source video (``<source>_<n>.avi``).
Clips from one source must never sit on both sides of a split, otherwise the
internal validation score is inflated and checkpoint selection is biased.

The official ``val`` folder is the test set and is never re-split.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

_SUFFIX = re.compile(r"_\d+$")

FIT = "fit"
INTERNAL_VAL = "internal_val"
TEST = "test"


class SplitError(ValueError):
    """Raised when a split cannot be built or fails validation."""


def source_video_id(rel_path: str) -> str:
    """Source video key: file stem with the trailing ``_<n>`` clip number removed."""
    name = rel_path.replace("\\", "/").split("/")[-1]
    stem = name.rsplit(".", 1)[0]
    return _SUFFIX.sub("", stem)


def build_grouped_split(
    entries: Iterable[Mapping],
    holdout_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign every manifest entry to ``fit``, ``internal_val`` or ``test``.

    ``entries`` need ``rel_path``, ``target`` and ``split``. Entries whose split is
    ``test`` stay in test; all others (``train``/``val_carved``) are pooled and a
    ~``holdout_fraction`` share of source videos is held out, stratified by label.
    """
    if not 0.0 < holdout_fraction < 0.5:
        raise SplitError(f"holdout_fraction must be in (0, 0.5), got {holdout_fraction}")
    entries = list(entries)
    assignment: Dict[str, str] = {}
    pool: List[Mapping] = []
    for e in entries:
        if e["split"] == "test":
            assignment[e["rel_path"]] = TEST
        else:
            pool.append(e)
    if not pool:
        raise SplitError("no training entries to split")

    y = np.asarray([int(e["target"]) for e in pool])
    groups = np.asarray([source_video_id(e["rel_path"]) for e in pool])
    n_splits = max(2, round(1.0 / holdout_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, hold_idx = next(sgkf.split(np.zeros(len(pool)), y, groups))
    hold = set(hold_idx.tolist())
    for i, e in enumerate(pool):
        assignment[e["rel_path"]] = INTERNAL_VAL if i in hold else FIT
    validate_split(entries, assignment)
    return assignment


def validate_split(entries: Iterable[Mapping], assignment: Mapping[str, str]) -> None:
    """Raise if any source video spans two splits or an entry is unassigned."""
    by_source: Dict[str, set] = {}
    for e in entries:
        part = assignment.get(e["rel_path"])
        if part is None:
            raise SplitError(f"entry not assigned: {e['rel_path']}")
        by_source.setdefault(source_video_id(e["rel_path"]), set()).add(part)
    leaks = sorted(s for s, parts in by_source.items() if len(parts) > 1)
    if leaks:
        raise SplitError(f"{len(leaks)} source videos span splits, e.g. {leaks[:3]}")


def split_summary(entries: Iterable[Mapping], assignment: Mapping[str, str]) -> Dict[str, Dict[str, int]]:
    """Clip, positive and source-video counts per split."""
    clips: Counter = Counter()
    pos: Counter = Counter()
    sources: Dict[str, set] = {}
    for e in entries:
        part = assignment[e["rel_path"]]
        clips[part] += 1
        pos[part] += int(e["target"])
        sources.setdefault(part, set()).add(source_video_id(e["rel_path"]))
    return {
        part: {"clips": clips[part], "violent": pos[part], "source_videos": len(sources[part])}
        for part in sorted(clips)
    }
