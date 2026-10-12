"""EmbeddingGemma 2 inference with ONNX Runtime on CPU.

Text and images go through the same language-model graph and land in the
same 768-dim space. Image preprocessing is a numpy port of the reference
Gemma 4 image processor: aspect-preserving resize to a multiple of 48 px
within the token budget, 16x16 patches in [0, 1], (x, y) patch positions.
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from .config import (
    BOI_ID,
    BOS_ID,
    DEFAULT_IMAGE_BUDGET,
    DOC_PREFIX_TEMPLATE,
    EMBED_DIM,
    EOI_ID,
    EOS_ID,
    HIDDEN_DIM,
    IMAGE_BUDGETS,
    IMAGE_ID,
    MAX_SEQ_TOKENS,
    PAD_ID,
    PATCH_SIZE,
    POOLING_KERNEL,
    QUERY_PREFIX,
)
from .errors import ModelLoadError

# Images with more pixels than this are not decoded. A JPEG is decoded at a
# reduced scale (draft mode), so it may be larger; other formats need the full
# raster in memory (4 bytes per pixel for RGBA).
MAX_IMAGE_PIXELS = 64_000_000
MAX_JPEG_PIXELS = 150_000_000
# WebP cannot be decoded at reduced scale and its decoder needs about 16 bytes
# per pixel (1 GB at 64M pixels).
MAX_WEBP_PIXELS = 25_000_000
# Larger images are scaled down to about this many pixels right after
# decoding, before any conversion copies them. The largest token budget uses
# about 2.6M pixels.
DOWNSCALE_PIXELS = 4_000_000
# Pillow's own decompression-bomb guard (it refuses twice this).
Image.MAX_IMAGE_PIXELS = MAX_JPEG_PIXELS


class ImageTooLarge(ValueError):
    """The image has too many pixels to decode safely."""


def runtime_versions() -> str:
    import onnxruntime
    import tokenizers

    return f"onnxruntime {onnxruntime.__version__}, tokenizers {tokenizers.__version__}"


def _load_error(what: str, ex: Exception) -> ModelLoadError:
    try:
        versions = runtime_versions()
    except Exception:
        versions = "unknown versions"
    return ModelLoadError(
        f"Could not load {what}: {type(ex).__name__}: {ex}\n"
        f"Installed: {versions}. anygrep needs onnxruntime>=1.23 and tokenizers>=0.20;\n"
        "upgrade them (for example `pipx upgrade anygrep` or `pip install -U onnxruntime "
        "tokenizers`)."
    )


class Embedder(Protocol):
    """What the indexer and searcher need from a model."""

    image_budget: int

    def token_offsets(self, text: str) -> list[tuple[int, int]]: ...

    def embed_documents(self, texts: Sequence[str], titles: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_image(self, image: Image.Image) -> np.ndarray: ...


# ---------------------------------------------------------------- images
def target_size(height: int, width: int, budget: int) -> tuple[int, int]:
    """Largest (h, w), multiples of 48, with at most budget*9 16-px patches."""
    if budget not in IMAGE_BUDGETS:
        raise ValueError(f"image token budget must be one of {IMAGE_BUDGETS}")
    max_patches = budget * POOLING_KERNEL**2
    target_px = max_patches * PATCH_SIZE**2
    factor = math.sqrt(target_px / (height * width))
    side = POOLING_KERNEL * PATCH_SIZE
    th = math.floor(factor * height / side) * side
    tw = math.floor(factor * width / side) * side
    max_side = (max_patches // POOLING_KERNEL**2) * side
    if th == 0 and tw == 0:
        raise ValueError("image too small")
    if th == 0:
        th = side
        tw = min(math.floor(width / height) * side, max_side)
    elif tw == 0:
        tw = side
        th = min(math.floor(height / width) * side, max_side)
    return th, tw


def load_image(path: str | Path) -> Image.Image:
    """Open an image: first frame, EXIF orientation applied, RGB on white.

    Raises ImageTooLarge for images with too many pixels, before decoding."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            im = Image.open(path)
    except Image.DecompressionBombError as e:
        raise ImageTooLarge(str(e)) from None
    with contextlib.suppress(EOFError):
        im.seek(0)
    w, h = im.size  # known from the header; nothing is decoded yet
    limit = {"JPEG": MAX_JPEG_PIXELS, "WEBP": MAX_WEBP_PIXELS}.get(im.format, MAX_IMAGE_PIXELS)
    if w * h > limit:
        raise ImageTooLarge(f"image too large ({w}x{h} pixels, limit {limit // 1_000_000}M)")
    if im.format == "JPEG":
        # Decode JPEGs at reduced scale when they are much larger than needed.
        im.draft("RGB", (2048, 2048))
    im = _downscale(im)
    im = ImageOps.exif_transpose(im) or im
    return to_rgb(im)


def _downscale(im: Image.Image) -> Image.Image:
    """Scale a large image down to about DOWNSCALE_PIXELS in its own mode, so
    the copies made later are small. The aspect ratio is kept."""
    w, h = im.size
    if w * h <= DOWNSCALE_PIXELS:
        return im
    s = math.sqrt(DOWNSCALE_PIXELS / (w * h))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    if im.mode in ("1", "P", "PA", "LA", "RGBA") and s < 0.5:
        # Converting these (to RGB, or to premultiplied alpha for filtering)
        # copies the whole raster at up to 4 bytes per pixel: first take every
        # n-th pixel, down to twice the final size, which copies nothing.
        mid = (max(1, round(w * 2 * s)), max(1, round(h * 2 * s)))
        im = im.resize(mid, Image.Resampling.NEAREST)
    if im.mode in ("1", "P", "PA"):  # these would be resized without filtering
        im = im.convert("RGBA" if im.mode == "PA" or "transparency" in im.info else "RGB")
    return im.resize(size, Image.Resampling.BICUBIC, reducing_gap=2.0)


def to_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


def image_to_patches(im: Image.Image, budget: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (pixel_values [P, 768], positions [P, 2], n_soft_tokens)."""
    w, h = im.size
    th, tw = target_size(h, w, budget)
    if (tw, th) != (w, h):
        im = im.resize((tw, th), Image.Resampling.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)
    ph, pw = th // PATCH_SIZE, tw // PATCH_SIZE
    patches = (
        arr.reshape(ph, PATCH_SIZE, pw, PATCH_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(ph * pw, PATCH_SIZE * PATCH_SIZE * 3)
    )
    gx, gy = np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy")
    pos = np.stack([gx, gy], axis=-1).reshape(ph * pw, 2).astype(np.int64)
    return np.ascontiguousarray(patches), pos, (ph * pw) // POOLING_KERNEL**2


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


# ---------------------------------------------------------------- ONNX model
class OnnxEmbedder:
    """EmbeddingGemma 2 int8 ONNX model (text graph + optional vision encoder)."""

    def __init__(
        self, model_dir: Path, threads: int, image_budget: int = DEFAULT_IMAGE_BUDGET
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # ORT_DISABLE_TELEMETRY (set in __init__.py) stops the Linux build from
        # sending usage events. The Windows build does not read it; it logs
        # them through ETW, which this call turns off.
        if os.environ.get("ORT_DISABLE_TELEMETRY") == "1":
            with contextlib.suppress(Exception):
                ort.disable_telemetry_events()
        self._ort = ort
        self.model_dir = Path(model_dir)
        self.threads = threads
        self.image_budget = image_budget
        # Asked between model runs of a text batch (set by the indexer).
        self.should_stop: Callable[[], bool] | None = None
        try:
            self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            # Text like "<|image|>" or "<eos>" in a note is plain text, not a
            # control token (an image placeholder without image features makes
            # the model fail).
            self.tokenizer.encode_special_tokens = True
        except Exception as ex:
            raise _load_error("the tokenizer", ex) from ex
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self._lm = self._session("onnx/model_quantized.onnx")
        self._vision = None
        self._empty = np.zeros((0, HIDDEN_DIM), np.float32)

    def _session(self, rel: str):
        so = self._ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
        # The vision encoder allocates large attention buffers; without the
        # arena they are returned to the OS and resident memory stays low.
        so.enable_cpu_mem_arena = False
        so.log_severity_level = 3
        try:
            return self._ort.InferenceSession(
                str(self.model_dir / rel), so, providers=["CPUExecutionProvider"]
            )
        except Exception as ex:
            raise _load_error(rel, ex) from ex

    # -- tokens
    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).offsets)

    def _ids(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return [BOS_ID, *ids[: MAX_SEQ_TOKENS - 2], EOS_ID]

    def _run_lm(self, ids: np.ndarray, mask: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        out = self._lm.run(
            ["sentence_embedding"],
            {
                "input_ids": ids,
                "attention_mask": mask,
                "image_features": image_features,
                "video_features": self._empty,
                "audio_features": self._empty,
            },
        )[0]
        return _normalize(out)

    def _embed_ids(self, seqs: list[list[int]], max_batch_tokens: int = 4096) -> np.ndarray:
        out = np.zeros((len(seqs), EMBED_DIM), np.float32)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        i = 0
        while i < len(order):
            if i and self.should_stop is not None and self.should_stop():
                raise KeyboardInterrupt  # a background stop request
            j = i + 1
            # Grow the batch while the padded size stays within budget.
            while j < len(order) and len(seqs[order[j]]) * (j - i + 1) <= max_batch_tokens:
                j += 1
            idx = order[i:j]
            L = max(len(seqs[k]) for k in idx)
            ids = np.full((len(idx), L), PAD_ID, np.int64)
            mask = np.zeros((len(idx), L), np.int64)
            for r, k in enumerate(idx):
                s = seqs[k]
                ids[r, : len(s)] = s
                mask[r, : len(s)] = 1
            out[idx] = self._run_lm(ids, mask, self._empty)
            i = j
        return out

    # -- public
    def embed_documents(self, texts: Sequence[str], titles: Sequence[str]) -> np.ndarray:
        seqs = [
            self._ids(DOC_PREFIX_TEMPLATE.format(title=_clean_title(t)) + x)
            for x, t in zip(texts, titles, strict=True)
        ]
        return self._embed_ids(seqs)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_ids([self._ids(QUERY_PREFIX + text)])[0]

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self._embed_patches([image_to_patches(image, self.image_budget)])[0]

    def _embed_patches(self, items: list[tuple[np.ndarray, np.ndarray, int]]) -> np.ndarray:
        """Images with the same patch grid, as one batch (the indexer passes
        one image at a time)."""
        if self._vision is None:
            self._vision = self._session("onnx/vision_encoder_quantized.onnx")
        pv = np.stack([x[0] for x in items])
        pos = np.stack([x[1] for x in items])
        n_tok = items[0][2]
        feats = self._vision.run(None, {"pixel_values": pv, "pixel_position_ids": pos})[0]
        if feats.shape[0] != n_tok * len(items):
            raise RuntimeError(
                f"vision encoder returned {feats.shape[0]} tokens, expected {n_tok * len(items)}"
            )
        ids = np.array([[BOS_ID, BOI_ID, *([IMAGE_ID] * n_tok), EOI_ID, EOS_ID]], np.int64)
        feats = feats.astype(np.float32)
        # The language model runs one image at a time: batched, its vectors
        # differ in the last bits (the vision encoder's do not), and it is
        # 14 percent of the time.
        return np.stack(
            [
                self._run_lm(ids, np.ones_like(ids), feats[i * n_tok : (i + 1) * n_tok])[0]
                for i in range(len(items))
            ]
        )


def embedder_factory(threads: int, quiet: bool = True) -> Callable[[bool], OnnxEmbedder]:
    """factory(vision) -> embedder. The model is loaded on the first call and
    reused; a later call that needs the vision encoder makes sure its files
    exist (downloading them unless offline; ModelUnavailableError otherwise).
    `quiet` hides the download messages and progress bar."""
    from .model import ensure_model

    cache: dict = {}

    def factory(vision: bool) -> OnnxEmbedder:
        if "emb" in cache and (cache["vision"] or not vision):
            return cache["emb"]
        d = ensure_model(vision=vision, quiet=quiet)
        if "emb" not in cache:
            cache["emb"] = OnnxEmbedder(d, threads=threads)
        cache["vision"] = vision
        return cache["emb"]

    return factory


def _clean_title(title: str) -> str:
    t = " ".join(title.replace("|", " ").split())
    return t or "none"
