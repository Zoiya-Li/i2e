"""Load omnimatte.ir.json and render the four edit classes to prove the layered doc is
editable: (1) recolor hero, (2) relabel text, (3) delete a secondary object, (4) free-move
the hero with its smoke/shadow following. Outputs PNGs under work/poster/demo/."""
import json, sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IR = ROOT / "work/poster/omnimatte.ir.json"
DEMO = ROOT / "work/poster/demo"
DEMO.mkdir(parents=True, exist_ok=True)


def recolor_hue(rgba, deg):
    """Rotate hue of the RGB channels by `deg` degrees; alpha untouched."""
    rgb = rgba[..., :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(deg / 2)) % 180          # OpenCV hue is 0..179
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.dstack([out, rgba[..., 3]])


def paste_rgba(base_rgb, rgba, x, y):
    """Alpha-composite an RGBA layer onto base_rgb at top-left (x, y)."""
    H, W = base_rgb.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return base_rgb
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[..., 3:4].astype(np.float32)) / 255.0
    base_rgb[y0:y1, x0:x1] = (sub[..., :3] * a + base_rgb[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
    return base_rgb


def _load(ir):
    plate = np.array(Image.open(ir["plate"]).convert("RGB"))
    layers = []
    for L in sorted(ir["layers"], key=lambda d: d["z"], reverse=True):  # back->front
        layers.append((L, np.array(Image.open(L["asset"]).convert("RGBA"))))
    return plate, layers


def compose(ir, recolor=None, hide=None, move=None):
    """recolor: {id:deg}; hide: set(ids); move: {id:(dx,dy)}."""
    recolor = recolor or {}
    hide = hide or set()
    move = move or {}
    plate, layers = _load(ir)
    canvas = plate.copy()
    for L, rgba in layers:
        if L["id"] in hide:
            continue
        rg = recolor_hue(rgba, recolor[L["id"]]) if L["id"] in recolor else rgba
        dx, dy = move.get(L["id"], (0, 0))
        canvas = paste_rgba(canvas, rg, L["transform"]["x"] + dx, L["transform"]["y"] + dy)
    return canvas


def hero_ids(ir):
    return [L["id"] for L in ir["layers"]
            if any(k in L["name"].lower() for k in ("tub", "scoop", "cup"))]


def main():
    ir = json.load(open(IR))
    Image.fromarray(compose(ir)).save(DEMO / "00_original_recomposite.png")
    Image.fromarray(compose(ir, recolor={i: 150 for i in hero_ids(ir)})).save(DEMO / "01_recolor_hero.png")
    sec = next((L["id"] for L in ir["layers"] if "bottle" in L["name"].lower()), None)
    if sec:
        Image.fromarray(compose(ir, hide={sec})).save(DEMO / "02_delete_bottle.png")
    Image.fromarray(compose(ir, move={i: (-300, -150) for i in hero_ids(ir)})).save(DEMO / "03_move_hero.png")
    print("wrote demo PNGs ->", DEMO)
    print("NOTE relabel (#04) is a text-layer swap handled in the interactive editor.")


if __name__ == "__main__":
    main()
