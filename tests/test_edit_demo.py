"""Unit tests for edit demo helpers: recolor_hue + paste_rgba."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_recolor_hue_shifts_color_keeps_alpha():
    from work.edit_demo import recolor_hue
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 1] = 200      # opaque green
    rgba[..., 3] = 255
    out = recolor_hue(rgba, deg=150)
    assert out[..., 3].min() == 255                    # alpha preserved
    assert not np.array_equal(out[..., :3], rgba[..., :3])  # hue changed


def test_paste_rgba_respects_alpha_and_offset():
    from work.edit_demo import paste_rgba
    base = np.zeros((10, 10, 3), np.uint8)
    lay = np.zeros((4, 4, 4), np.uint8)
    lay[..., 0] = 255       # opaque red
    lay[..., 3] = 255
    out = paste_rgba(base, lay, x=3, y=3)
    assert tuple(out[4, 4]) == (255, 0, 0)             # pasted at offset
    assert tuple(out[0, 0]) == (0, 0, 0)               # untouched elsewhere


def test_paste_rgba_clips_to_bounds():
    from work.edit_demo import paste_rgba
    base = np.zeros((10, 10, 3), np.uint8)
    lay = np.full((8, 8, 4), (255, 0, 0, 255), np.uint8)
    # paste at x=7, y=7 — only a 3x3 corner lands on the 10x10 canvas
    out = paste_rgba(base, lay, x=7, y=7)
    # should not crash and should only paste the visible 3x3 corner
    assert out.shape == (10, 10, 3)
    assert out[9, 9].sum() > 0   # corner pasted
    assert out[0, 0].sum() == 0  # far corner untouched


def test_paste_rgba_partial_alpha():
    from work.edit_demo import paste_rgba
    base = np.full((5, 5, 3), 100, np.uint8)
    lay = np.full((2, 2, 4), (200, 0, 0, 128), np.uint8)  # 50% alpha red
    out = paste_rgba(base, lay, x=0, y=0)
    # 50% blend: 200*0.5 + 100*0.5 = 150
    assert abs(int(out[0, 0, 0]) - 150) < 3
