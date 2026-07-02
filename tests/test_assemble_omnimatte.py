"""Unit tests for omnimatte IR assembly."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_ir_shape(tmp_path):
    from work.assemble_omnimatte import build_ir
    layers = {"W": 100, "H": 200, "layers": [
        {"id": "raster-1", "name": "cup", "z": 0, "asset": "a.png",
         "x": 10, "y": 20, "w": 30, "h": 40}]}
    ir = build_ir(layers, plate="plate.png", texts=[{
        "id": "t1", "content": "风油精", "x": 5, "y": 5, "w": 20, "h": 8}])
    assert ir["plate"] == "plate.png"
    assert ir["canvas"] == {"w": 100, "h": 200}
    L = ir["layers"][0]
    assert L["transform"] == {"x": 10, "y": 20, "scale": 1.0, "rotation": 0.0}
    assert L["bbox"] == {"x": 10, "y": 20, "w": 30, "h": 40}
    assert ir["layers"][0]["z"] == 0
    assert ir["texts"][0]["content"] == "风油精"
