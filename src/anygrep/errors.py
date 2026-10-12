"""Exceptions raised by anygrep. They all derive from AnygrepError."""

from __future__ import annotations


class AnygrepError(Exception):
    """Base class of the errors anygrep raises on purpose."""


class IndexBusyError(AnygrepError, RuntimeError):
    """Another anygrep process (often a background index) is changing the
    index, so it cannot be updated now. Searching still works."""


class IndexMismatchError(AnygrepError, RuntimeError):
    """The file is not an anygrep index, or it was built by another version or
    with another model revision."""


class ModelUnavailableError(AnygrepError, RuntimeError):
    """The model files are missing and could not be downloaded."""


class ModelLoadError(AnygrepError, RuntimeError):
    """The model files exist but this onnxruntime/tokenizers cannot load them."""


class IndexAccessError(AnygrepError, OSError):
    """The index file cannot be opened or written: it is a folder, read-only,
    in a folder that cannot be written, or the disk is full."""
