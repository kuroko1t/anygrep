from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anygrep import cli, job

ROOT = Path(__file__).resolve().parents[1]


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "make_demo_corpus", ROOT / "scripts" / "make_demo_corpus.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load_demo_module()

_WORD = re.compile(r"\w+")


def _word_vec(w: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha1(w.lower().encode()).digest()[:4], "little")
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


class FakeEmbedder:
    """Bag-of-words hashing embedder: same words -> similar vectors.

    Tokens are whitespace/word pieces, so chunking can be tested without the
    real tokenizer. Images embed to the vector of their dominant colour name.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.image_budget = 140
        self.calls = {"docs": 0, "query": 0, "image": 0}
        self.fail_after: int | None = None  # raise KeyboardInterrupt after N embeds

    def _tick(self) -> None:
        if self.fail_after is not None:
            if self.fail_after <= 0:
                raise KeyboardInterrupt
            self.fail_after -= 1

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for w in _WORD.findall(text):
            v += _word_vec(w, self.dim)
        n = np.linalg.norm(v)
        return v / n if n else v

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]

    def embed_documents(self, texts: Sequence[str], titles: Sequence[str]) -> np.ndarray:
        self._tick()
        self.calls["docs"] += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.calls["query"] += 1
        return self._vec(text)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self._tick()
        self.calls["image"] += 1
        r, g, b = np.asarray(image.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)
        name = "red" if r > max(g, b) else "green" if g > b else "blue"
        return self._vec(f"{name} picture image")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    demo.write_pdf(path, pages)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "bread.md").write_text(
        "# Bread\n\nSourdough bread needs flour water salt and a starter.\n"
        "Bake the bread in a hot oven.\n",
        encoding="utf-8",
    )
    (root / "notes" / "bikes.txt").write_text(
        "Fixing a bicycle chain and pumping the tyres.\n", encoding="utf-8"
    )
    write_pdf(
        root / "manual.pdf",
        [["Router manual", "Plug in the router."], ["Wifi password", "Reset the wifi password."]],
    )
    Image.new("RGB", (200, 150), (220, 20, 20)).save(root / "red.png")
    Image.new("RGB", (200, 150), (20, 20, 220)).save(root / "blue.jpg")
    return root


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANYGREP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANYGREP_DB", raising=False)
    monkeypatch.delenv("ANYGREP_INDEX_BUDGET", raising=False)
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")  # an empty folder: searches name their scope
    emb = FakeEmbedder()
    monkeypatch.setattr(cli, "_make_embedder_factory", lambda args: lambda vision: emb)
    # A background job runs to the end inside the test, before spawn returns.
    monkeypatch.setattr(cli, "_prefetch_model", lambda vision, quiet: None)
    monkeypatch.setattr(job, "_detach_setup", lambda: None)
    monkeypatch.setattr(job, "spawn", sync_spawn)
    return emb


def sync_spawn(argv: list[str], log_path: Path, cwd: Path) -> int:
    """job.spawn for tests: run the worker in this process, output to the log."""
    hint = cli._DB_HINT
    try:
        with (
            open(log_path, "a", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            sync_spawn.codes.append(cli.main(argv[len(job.WORKER_PREFIX) :]))
    finally:
        cli._DB_HINT = hint
    sync_spawn.argvs.append(argv)
    return os.getpid()


sync_spawn.codes = []
sync_spawn.argvs = []


@pytest.fixture
def tty(monkeypatch):
    """stdout counts as a terminal (the human result view)."""

    def on() -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)

    return on


def jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]
