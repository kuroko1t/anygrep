"""A background index worker for tests: anygrep with the fake model, slow on
images ($ANYGREP_TEST_IMAGE_DELAY seconds each), so the real spawn, detach,
lock, log, state and stop paths run without the model files."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# This checkout's code, ahead of any installed anygrep (e.g. an editable
# install of another worktree), and the tests package for the fake model.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anygrep import cli
from tests.conftest import FakeEmbedder

DELAY = float(os.environ.get("ANYGREP_TEST_IMAGE_DELAY", "0.1"))


class SlowEmbedder(FakeEmbedder):
    def embed_image(self, image):
        time.sleep(DELAY)
        return super().embed_image(image)


def main() -> int:
    emb = SlowEmbedder()
    cli._make_embedder_factory = lambda args: lambda vision: emb
    try:
        return cli.main(sys.argv[1:])
    finally:
        print(f"image embeds: {emb.calls['image']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
