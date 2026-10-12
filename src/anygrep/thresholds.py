"""Decide which search results are weak matches.

Cosine ranges differ by the pair of query type and result type (a text query
scores lower against images than against text), so each pair has its own
minimum score.

A minimum score alone lets through the best results of queries that match
nothing in particular ("asdf", "hello", "..."): such queries score a little
higher against everything. So a result must also score a margin above the
query's reference level: its median score against a fixed set of varied
texts and pictures that ships with anygrep (reference.npz, built by
scripts/build_reference_set.py). The reference does not depend on what was
indexed, so a folder where most files are about the query's topic does not
hide its own matches.

The values were measured with scripts/calibrate_thresholds.py; see
docs/DESIGN.md for the tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np

from .config import DEFAULT_IMAGE_BUDGET, MODEL_REVISION

# {"<query type>-><result type>": {image budget: minimum score}}. Pairs where
# neither side is an image use the key 0 for every budget.
MIN_SCORES: dict[str, dict[int, float]] = {
    "text->text": {0: 0.62},
    "text->pdf": {0: 0.64},
    "text->image": {70: 0.63, 140: 0.62, 280: 0.62},
    "image->image": {70: 0.76, 140: 0.75, 280: 0.70},
    "image->text": {70: 0.62, 140: 0.62, 280: 0.55},
    "image->pdf": {70: 0.62, 140: 0.62, 280: 0.61},
}

# How far a result must score above the query's reference level, per pair.
MIN_MARGINS: dict[str, float] = {
    "text->text": 0.10,
    "text->pdf": 0.11,
    "text->image": 0.06,
    "image->image": 0.09,
    "image->text": 0.04,
    "image->pdf": 0.13,
}

REFERENCE_FILE = Path(__file__).with_name("reference.npz")


def min_score(query_kind: str, result_kind: str, budget: int = DEFAULT_IMAGE_BUDGET) -> float:
    """The minimum score for a `result_kind` result of a `query_kind` query."""
    table = MIN_SCORES.get(f"{query_kind}->{result_kind}")
    if not table:
        return 0.0
    if 0 in table:
        return table[0]
    return table[_nearest(table, budget)]


def min_margin(query_kind: str, result_kind: str) -> float:
    return MIN_MARGINS.get(f"{query_kind}->{result_kind}", 0.0)


def _nearest(budgets, budget: int) -> int:
    return min(budgets, key=lambda b: (abs(b - budget), b))


@cache
def _reference() -> dict[str, np.ndarray] | None:
    """The reference vectors, or None when they do not match the model."""
    try:
        with np.load(REFERENCE_FILE) as z:
            if str(z["model_revision"]) != MODEL_REVISION:
                return None
            return {k: z[k].astype(np.float32) for k in z.files if k != "model_revision"}
    except (OSError, KeyError, ValueError):
        return None


def reference_level(
    qvec: np.ndarray, result_kind: str, budget: int = DEFAULT_IMAGE_BUDGET
) -> float | None:
    """Median score of the query against the reference set for this result type."""
    ref = _reference()
    if ref is None:
        return None
    if result_kind == "image":
        budgets = [int(k.split("_")[1]) for k in ref if k.startswith("image_")]
        mat = ref[f"image_{_nearest(budgets, budget)}"]
    else:
        mat = ref["text"]
    return float(np.median(mat @ np.asarray(qvec, np.float32)))


def thresholds(
    qvec: np.ndarray | None,
    kinds: Sequence[str],
    query_kind: str = "text",
    budget: int = DEFAULT_IMAGE_BUDGET,
) -> dict[str, float]:
    """The score a result of each type needs to count as a strong match."""
    out = {}
    for kind in set(kinds):
        t = min_score(query_kind, kind, budget)
        level = None if qvec is None else reference_level(qvec, kind, budget)
        if level is not None:
            t = max(t, level + min_margin(query_kind, kind))
        out[kind] = t
    return out


def weak_mask(
    scores: np.ndarray,
    kinds: Sequence[str],
    query_kind: str = "text",
    budget: int = DEFAULT_IMAGE_BUDGET,
    qvec: np.ndarray | None = None,
) -> np.ndarray:
    """True for every row whose score counts as a weak match. Without `qvec`
    only the minimum scores apply."""
    kinds_arr = np.asarray(kinds)
    weak = np.zeros(len(scores), bool)
    present = np.unique(kinds_arr).tolist() if len(kinds_arr) else []
    for kind, t in thresholds(qvec, present, query_kind, budget).items():
        sel = kinds_arr == kind
        weak[sel] = scores[sel] < t
    return weak
