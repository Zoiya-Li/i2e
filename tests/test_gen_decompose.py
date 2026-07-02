"""Tests for the generative decomposition pipeline."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---- identify.py tests ----

def test_parse_response_clean_json():
    from work.gen_decompose.identify import _parse_response
    raw = json.dumps({"entities": [
        {"name": "tub", "category": "product", "visual_desc": "green tub",
         "gen_prompt": "green tub on white", "bbox_frac": {"x": 0.3, "y": 0.3, "w": 0.5, "h": 0.5},
         "z_order": 1},
    ]})
    ents = _parse_response(raw)
    assert len(ents) == 1
    assert ents[0]["name"] == "tub"


def test_parse_response_json_in_text():
    from work.gen_decompose.identify import _parse_response
    raw = 'Here are the entities I found:\n{"entities": [{"name": "leaf", "category": "decoration", "visual_desc": "mint leaf", "gen_prompt": "mint leaf", "bbox_frac": {"x":0.8,"y":0.7,"w":0.15,"h":0.2}, "z_order": 2}]}\nThat covers all entities.'
    ents = _parse_response(raw)
    assert len(ents) == 1
    assert ents[0]["name"] == "leaf"


def test_parse_response_bare_array():
    from work.gen_decompose.identify import _parse_response
    raw = '[{"name": "bg", "category": "background", "visual_desc": "green gradient", "gen_prompt": "green gradient bg", "bbox_frac": {"x":0,"y":0,"w":1,"h":1}, "z_order": 0}]'
    ents = _parse_response(raw)
    assert len(ents) == 1
    assert ents[0]["category"] == "background"


def test_identify_assigns_ids():
    from work.gen_decompose.identify import identify_entities
    mock_driver = MagicMock()
    mock_driver.analyze.return_value = json.dumps({"entities": [
        {"name": "a", "category": "product", "visual_desc": "", "gen_prompt": "",
         "bbox_frac": {"x":0,"y":0,"w":0.5,"h":0.5}, "z_order": 0},
        {"name": "b", "category": "decoration", "visual_desc": "", "gen_prompt": "",
         "bbox_frac": {"x":0.5,"y":0.5,"w":0.5,"h":0.5}, "z_order": 1},
    ]})
    ents = identify_entities("fake.png", driver=mock_driver)
    assert ents[0]["id"] == "entity-1"
    assert ents[1]["id"] == "entity-2"


# ---- generate_layers.py tests ----

def test_build_entity_prompt():
    from work.gen_decompose.generate_layers import _build_entity_prompt
    ent = {"name": "green tub", "gen_prompt": "dark green cylindrical tub",
           "visual_desc": "a green tub"}
    prompt = _build_entity_prompt(ent)
    assert "green tub" in prompt
    assert "white background" in prompt
    assert "Photorealistic" in prompt


def test_build_background_prompt():
    from work.gen_decompose.generate_layers import _build_background_prompt
    ent = {"name": "background", "gen_prompt": "dark green to black gradient with mist",
           "visual_desc": "green gradient"}
    prompt = _build_background_prompt(ent)
    assert "NO objects" in prompt
    assert "NO products" in prompt


def test_threshold_remove_bg():
    from work.gen_decompose.generate_layers import _threshold_remove_bg
    import tempfile
    # create a test image: white background + red square
    img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
    arr = np.array(img)
    arr[30:70, 30:70] = [255, 0, 0, 255]  # red square
    img = Image.fromarray(arr)

    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input.png"
        out = Path(tmp) / "output.png"
        img.save(inp)
        result = _threshold_remove_bg(str(inp), str(out))
        assert Path(result).exists()
        out_img = np.array(Image.open(out))
        # center should be opaque red
        assert out_img[50, 50, 0] == 255  # R
        assert out_img[50, 50, 3] == 255  # A
        # corner should be transparent (was white)
        assert out_img[5, 5, 3] == 0  # A=0


# ---- assemble.py tests ----

def test_assemble_creates_ir():
    from work.gen_decompose.assemble import assemble
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # create fake original
        orig = Image.new("RGB", (200, 400), (50, 100, 50))
        orig_path = tmp / "original.png"
        orig.save(orig_path)

        # create fake plate
        plate = Image.new("RGB", (200, 400), (30, 60, 30))
        plate_path = tmp / "plate_raw.png"
        plate.save(plate_path)

        # create fake entity (RGBA)
        layer = Image.new("RGBA", (100, 100), (255, 0, 0, 200))
        layer_path = tmp / "entity_layer.png"
        layer.save(layer_path)

        entities = [
            {"id": "bg", "name": "background", "category": "background",
             "visual_desc": "green", "gen_prompt": "green",
             "bbox_frac": {"x": 0, "y": 0, "w": 1, "h": 1}, "z_order": 0,
             "asset": str(plate_path)},
            {"id": "obj", "name": "product", "category": "product",
             "visual_desc": "red thing", "gen_prompt": "red thing",
             "bbox_frac": {"x": 0.2, "y": 0.3, "w": 0.5, "h": 0.4}, "z_order": 1,
             "asset": str(layer_path)},
        ]

        ir = assemble(entities, str(orig_path), str(tmp / "out"))

        assert ir["canvas"] == {"w": 200, "h": 400}
        assert len(ir["layers"]) == 1  # only the product layer
        assert ir["layers"][0]["name"] == "product"
        assert Path(ir["plate"]).exists()
        assert (tmp / "out" / "comparison.png").exists()


# ---- driver.py tests ----

def test_launch_chrome_importable():
    from work.gen_decompose.driver import launch_chrome
    assert callable(launch_chrome)


def test_gemini_webdriver_class_exists():
    from work.gen_decompose.driver import GeminiWebDriver
    wd = GeminiWebDriver()
    assert wd.port == 9222
    assert wd.timeout == 120
