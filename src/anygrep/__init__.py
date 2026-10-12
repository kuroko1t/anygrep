"""Local semantic search over notes, PDFs and images with one embedding model.

From Python:

    import anygrep

    with anygrep.Index() as idx:
        idx.update("~/notes")
        for hit in idx.search("when is the tax return due"):
            print(hit.path, hit.score)

See docs/python-api.md. The API may change before 1.0.
"""

import os

# Recent onnxruntime builds send usage events to Microsoft when a model is
# loaded and keep a device id under ~/.cache/Microsoft/DeveloperTools.
# onnxruntime reads this variable at startup, so it is set here, before any
# module imports onnxruntime; the background worker inherits it. A value the
# user set themselves is left alone.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"

# These modules import neither numpy nor onnxruntime: `import anygrep` stays cheap.
from .api import Hit, Index, UpdateResult
from .errors import (
    AnygrepError,
    IndexAccessError,
    IndexBusyError,
    IndexMismatchError,
    ModelLoadError,
    ModelUnavailableError,
)

__all__ = [
    "AnygrepError",
    "Hit",
    "Index",
    "IndexAccessError",
    "IndexBusyError",
    "IndexMismatchError",
    "ModelLoadError",
    "ModelUnavailableError",
    "UpdateResult",
    "__version__",
]
