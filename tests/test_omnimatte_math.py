"""Unit tests for omnimatte_math — delta_alpha + build_layer."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_delta_alpha_zero_when_no_change():
    from work.lib.omnimatte_math import delta_alpha
    a = np.full((8, 8, 3), 100, np.uint8)
    assert delta_alpha(a, a.copy()).max() == 0


def test_delta_alpha_full_where_object_vanished():
    from work.lib.omnimatte_math import delta_alpha
    before = np.zeros((8, 8, 3), np.uint8)
    after = before.copy()
    before[2:6, 2:6] = 255          # bright object present in `before`, gone in `after`
    al = delta_alpha(before, after, smooth_sigma=0)  # uint8 HxW, no smoothing on clean 8x8
    assert al.shape == (8, 8)
    assert al[3, 3] > 200            # object center -> high alpha
    assert al[0, 0] == 0             # untouched -> zero alpha


def test_delta_alpha_smooths_speckle():
    from work.lib.omnimatte_math import delta_alpha
    before = np.zeros((16, 16, 3), np.uint8)
    after = before.copy()
    before[8, 8] = 30                # tiny low-contrast 1px change
    al = delta_alpha(before, after, smooth_sigma=1.0, thresh=0.15)
    assert al.max() == 0             # below threshold -> removed as speckle


def test_build_layer_clamps_to_dilated_bbox():
    from work.lib.omnimatte_math import build_layer
    original = np.full((20, 20, 3), 50, np.uint8)
    original[5:10, 5:10] = 200                     # the object pixels
    before = original.copy()
    after = original.copy()
    after[5:10, 5:10] = 50                          # object removed
    after[18, 18] = 60                              # stray far-away change (different obj)
    rgba, (x0, y0, x1, y1) = build_layer(original, before, after,
                                         obj_bbox=(5, 5, 10, 10), clamp_pad=3)
    assert rgba.shape[2] == 4
    # crop is within the clamped region (does not span to (18,18))
    assert x1 <= 13 and y1 <= 13
