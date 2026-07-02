"""Tests for gemini_provider — mostly unit-level (no Chrome needed).
Integration test (test_extract_real) is skipped unless Chrome CDP is running."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_gemini_web_provider_is_registered():
    """get_provider('gemini-web') should return a GeminiWebProvider instance."""
    from extractor.providers import get_provider
    p = get_provider("gemini-web")
    assert p.name == "gemini-web"
    assert "gemini" in p.model_version


def test_parse_response_and_normalize_bbox():
    """GeminiWebProvider.extract() should parse JSON and normalize fraction bboxes."""
    from extractor.gemini_provider import GeminiWebProvider, _GeminiWebDriver
    from unittest.mock import patch, MagicMock

    fake_response = json.dumps({"elements": [
        {"type": "background", "name": "bg", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
         "confidence": 0.9, "text": None, "raster": None, "logo": None,
         "vector": None, "children": None},
        {"type": "text", "name": "headline", "bbox": {"x": 0.05, "y": 0.15, "w": 0.7, "h": 0.12},
         "confidence": 0.85,
         "text": {"content": "Hello", "font_family": None, "font_size_px": None,
                  "color": "#FFF", "align": "left", "lang": "en"},
         "raster": None, "logo": None, "vector": None, "children": None},
    ]})

    prov = GeminiWebProvider()
    # mock image size: 1000x2000
    mock_img = MagicMock()
    mock_img.size = (1000, 2000)

    with patch.object(prov, '_get_wd') as mock_wd, \
         patch('PIL.Image.open', return_value=mock_img):
        wd = MagicMock()
        wd.analyze.return_value = fake_response
        mock_wd.return_value = wd

        els = prov.extract("fake.png")

    assert len(els) == 2
    # background: (0,0,1,1) → (0,0,1000,2000)
    bg = els[0]
    assert bg["bbox"]["w"] == 1000.0
    assert bg["bbox"]["h"] == 2000.0
    # headline: (0.05,0.15,0.7,0.12) → (50,300,700,240)
    hl = els[1]
    assert hl["bbox"]["x"] == 50.0
    assert hl["bbox"]["y"] == 300.0
    assert hl["bbox"]["w"] == 700.0
    assert hl["bbox"]["h"] == 240.0


def test_pixel_bbox_unchanged():
    """If bbox values look like pixels (>1.5), leave them as-is."""
    from extractor.gemini_provider import GeminiWebProvider
    from unittest.mock import patch, MagicMock

    fake_response = json.dumps({"elements": [
        {"type": "text", "name": "t", "bbox": {"x": 50, "y": 100, "w": 200, "h": 30},
         "confidence": 0.9, "text": None, "raster": None, "logo": None,
         "vector": None, "children": None},
    ]})
    mock_img = MagicMock()
    mock_img.size = (800, 600)
    prov = GeminiWebProvider()
    with patch.object(prov, '_get_wd') as mock_wd, \
         patch('PIL.Image.open', return_value=mock_img):
        wd = MagicMock()
        wd.analyze.return_value = fake_response
        mock_wd.return_value = wd
        els = prov.extract("fake.png")

    assert els[0]["bbox"]["x"] == 50.0  # unchanged
    assert els[0]["bbox"]["w"] == 200.0


def test_launch_chrome_function_exists():
    """Sanity: launch_chrome is importable."""
    from extractor.gemini_provider import launch_chrome
    assert callable(launch_chrome)
