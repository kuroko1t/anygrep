from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anygrep.embedder import image_to_patches, load_image, target_size


@pytest.mark.parametrize("budget", [70, 140, 280])
@pytest.mark.parametrize(
    "hw",
    [(768, 1024), (1024, 768), (500, 500), (40, 2000), (3000, 60), (32, 8000), (8000, 32)],
)
def test_target_size_within_budget(budget: int, hw: tuple[int, int]) -> None:
    h, w = hw
    th, tw = target_size(h, w, budget)
    assert th % 48 == 0 and tw % 48 == 0
    assert th > 0 and tw > 0
    assert (th // 16) * (tw // 16) <= budget * 9


def test_target_size_rejects_unknown_budget() -> None:
    for budget in (100, 560):  # 560 and 1120 have no calibrated cutoffs
        with pytest.raises(ValueError):
            target_size(100, 100, budget)


def test_patches_shape_and_positions() -> None:
    im = Image.new("RGB", (1024, 768), (255, 0, 0))
    pv, pos, n = image_to_patches(im, 140)
    assert pv.shape[1] == 768 and pv.dtype == np.float32
    assert pv.shape[0] == pos.shape[0] == n * 9
    assert pos[0].tolist() == [0, 0] and pos[1].tolist() == [1, 0]
    # red channel is 1.0, others 0.0, interleaved per pixel (R, G, B, R, ...)
    assert np.allclose(pv[0, 0::3], 1.0) and np.allclose(pv[0, 1::3], 0.0)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    im = Image.new("RGB", (300, 100), (0, 128, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    p = tmp_path / "rot.jpg"
    im.save(p, exif=exif.tobytes())
    assert load_image(p).size == (100, 300)


def test_transparency_is_composited_on_white(tmp_path: Path) -> None:
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    p = tmp_path / "t.png"
    im.save(p)
    out = load_image(p)
    assert out.mode == "RGB"
    assert out.getpixel((10, 10)) == (255, 255, 255)


def test_gif_first_frame(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (64, 64), c) for c in [(255, 0, 0), (0, 0, 255)]]
    p = tmp_path / "a.gif"
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=100)
    r, _g, b = load_image(p).getpixel((5, 5))
    assert r > 200 and b < 50


def test_image_over_the_pixel_limit_is_refused_before_decoding(tmp_path: Path, monkeypatch) -> None:
    from anygrep import embedder

    monkeypatch.setattr(embedder, "MAX_IMAGE_PIXELS", 10_000)
    p = tmp_path / "big.png"
    Image.new("RGBA", (200, 100)).save(p)
    decoded = []
    monkeypatch.setattr(Image.Image, "load", lambda self: decoded.append(1))
    with pytest.raises(embedder.ImageTooLarge):
        load_image(p)
    assert not decoded


def test_pillow_bomb_guard_is_reported_as_too_large(tmp_path: Path, monkeypatch) -> None:
    from anygrep import embedder

    p = tmp_path / "big.jpg"
    Image.new("RGB", (300, 300)).save(p)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # Pillow refuses > 2000 px
    with pytest.raises(embedder.ImageTooLarge):
        load_image(p)


@pytest.mark.parametrize("mode", ["RGBA", "P", "L"])
def test_large_image_is_scaled_down_before_conversion(tmp_path: Path, monkeypatch, mode) -> None:
    from anygrep import embedder

    monkeypatch.setattr(embedder, "DOWNSCALE_PIXELS", 10_000)
    im = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    im.paste((200, 0, 0, 255), (0, 0, 200, 300))
    p = tmp_path / "big.png"
    im.convert(mode).save(p) if mode != "RGBA" else im.save(p)
    out = load_image(p)
    assert out.mode == "RGB"
    assert 110 <= out.size[0] <= 120 and 84 <= out.size[1] <= 90
    if mode == "RGBA":
        assert out.getpixel((100, 40)) == (255, 255, 255)  # transparent: white
        assert out.getpixel((10, 40))[0] > 150


@pytest.mark.parametrize("mode", ["RGBA", "P", "1", "LA"])
def test_large_palette_or_alpha_image_is_not_converted_at_full_size(
    tmp_path: Path, monkeypatch, mode
) -> None:
    from anygrep import embedder

    monkeypatch.setattr(embedder, "DOWNSCALE_PIXELS", 10_000)
    im = Image.new("RGB", (800, 600), (255, 255, 255))
    im.paste((0, 0, 0), (0, 0, 400, 600))
    p = tmp_path / "big.png"
    im.convert(mode).save(p)
    sizes = []
    convert = Image.Image.convert

    def spy(self, *a, **k):
        sizes.append(self.size[0] * self.size[1])
        return convert(self, *a, **k)

    monkeypatch.setattr(Image.Image, "convert", spy)
    out = load_image(p)
    assert out.mode == "RGB" and 110 <= out.size[0] <= 120
    assert max(sizes) <= 4 * 10_000 * 1.1  # at most twice the final size, per side
    assert out.getpixel((10, 40))[0] < 60 and out.getpixel((100, 40))[0] > 200


def test_webp_has_a_lower_pixel_limit(tmp_path: Path, monkeypatch) -> None:
    from anygrep import embedder

    monkeypatch.setattr(embedder, "MAX_WEBP_PIXELS", 10_000)
    p = tmp_path / "big.webp"
    Image.new("RGB", (200, 100)).save(p)
    with pytest.raises(embedder.ImageTooLarge, match="limit"):
        load_image(p)
    q = tmp_path / "big.png"
    Image.new("RGB", (200, 100)).save(q)
    assert load_image(q).size == (200, 100)
