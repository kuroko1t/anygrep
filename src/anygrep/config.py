"""Paths, environment variables and model constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs

APP_NAME = "anygrep"

# Model: EmbeddingGemma 2 (Google, Apache-2.0), int8 ONNX export by onnx-community.
MODEL_REPO = "onnx-community/embeddinggemma-2-ONNX"
MODEL_REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
MODEL_NAME = "google/embeddinggemma-2 (onnx int8)"


@dataclass(frozen=True)
class ModelFile:
    path: str
    size: int
    sha256: str
    vision: bool = False


MODEL_FILES: tuple[ModelFile, ...] = (
    ModelFile(
        "tokenizer.json",
        32170510,
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    ),
    ModelFile(
        "onnx/model_quantized.onnx",
        495165,
        "d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953",
    ),
    ModelFile(
        "onnx/model_quantized.onnx_data",
        313724928,
        "278a7ff1248c3618e4bd11a607fc54f7bdc7778854230f3956d3f86bd9db4f3b",
    ),
    ModelFile(
        "onnx/vision_encoder_quantized.onnx",
        162495,
        "bb0de2df53a2448a32dc7908a187c168c8afd514d4d6f674f7f46024875fa4e3",
        vision=True,
    ),
    ModelFile(
        "onnx/vision_encoder_quantized.onnx_data",
        195228672,
        "3dabd69c0a36e9a8771ad82030dde74daa5a0e02b7047a5d3f3382b1137bab89",
        vision=True,
    ),
)

# Token ids of the Gemma tokenizer used by EmbeddingGemma 2.
PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
BOI_ID = 255999  # <|image>
IMAGE_ID = 258880  # <|image|>
EOI_ID = 258882  # <image|>

EMBED_DIM = 768
HIDDEN_DIM = 512
PATCH_SIZE = 16
POOLING_KERNEL = 3
# Token budgets per image. The weak-match cutoffs (thresholds.py) are
# calibrated for these three only.
IMAGE_BUDGETS = (70, 140, 280)
DEFAULT_IMAGE_BUDGET = 140

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX_TEMPLATE = "title: {title} | text: "

# Text chunking (in model tokens, excluding the document prefix).
CHUNK_TOKENS = 384
CHUNK_OVERLAP = 48
MAX_SEQ_TOKENS = 2048

# Processing generation: raised by one whenever a change to anygrep makes it
# turn some kind of file into different chunks or vectors with the same model
# (text extraction, chunking, the document prompt, image preprocessing).
# Each file row records the generation it was processed with, and the next
# update processes again the files of the kinds a later generation changed
# (GENERATION_CHANGES). It is separate from the schema version (the layout
# of the index) and from the model revision (a different vector space, which
# is never mixed into an index). Policy: docs/DESIGN.md.
PROCESSING_GENERATION = 1
# {generation: the file kinds it changed}. Generation 1 is everything before
# generations were recorded; rows without one count as generation 1.
GENERATION_CHANGES: dict[int, frozenset[str]] = {}


def required_generation(kind: str) -> int:
    """The oldest generation whose rows of this kind are still current."""
    return max(
        [1]
        + [
            g
            for g, kinds in GENERATION_CHANGES.items()
            if kind in kinds and g <= PROCESSING_GENERATION
        ]
    )


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def data_dir() -> Path:
    return _env_path("ANYGREP_HOME") or Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))


def default_index_path() -> Path:
    return _env_path("ANYGREP_DB") or data_dir() / "index.sqlite"


# Seconds of indexing a search may do before it searches (section 4 of the CLI
# spec): new or changed files under the searched folder are indexed first when
# the estimate fits, otherwise anygrep searches what is indexed.
DEFAULT_INDEX_BUDGET = 30.0


def default_index_budget() -> float:
    """$ANYGREP_INDEX_BUDGET or the default; ValueError when it is not a
    number of seconds >= 0."""
    env = os.environ.get("ANYGREP_INDEX_BUDGET", "").strip()
    if not env:
        return DEFAULT_INDEX_BUDGET
    v = float(env)
    if not v >= 0:  # also rejects nan
        raise ValueError(env)
    return v


def model_dir() -> Path:
    """Directory holding tokenizer.json and onnx/*.onnx(_data)."""
    override = _env_path("ANYGREP_MODEL_DIR")
    if override:
        return override
    base = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)) / "models"
    return base / MODEL_REPO.replace("/", "--") / MODEL_REVISION[:12]


def offline() -> bool:
    return any(
        os.environ.get(k, "").lower() in ("1", "true", "yes")
        for k in ("ANYGREP_OFFLINE", "HF_HUB_OFFLINE")
    )


def default_threads() -> int:
    """$ANYGREP_THREADS, or half the CPU threads up to 8. Raises ValueError
    when $ANYGREP_THREADS is set but is not a whole number of at least 1."""
    env = os.environ.get("ANYGREP_THREADS", "").strip()
    if env:
        if not (env.isdigit() and int(env) > 0):
            raise ValueError(env)
        return int(env)
    n = os.cpu_count() or 2
    # Physical cores are usually half the logical count; leave headroom.
    return max(1, min(8, n // 2))


def default_bg_threads() -> int:
    """Threads of a background index: half the default, 1 to 4, so the
    computer stays usable while it runs. Raises ValueError like
    default_threads()."""
    return max(1, min(4, default_threads() // 2))


def embed_device() -> str:
    """Where the model runs. CPU only for now; the one place to change."""
    return "cpu"
