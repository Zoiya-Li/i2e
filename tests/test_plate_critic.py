"""Unit tests for the plate critic predicate (pure logic, no GPU)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_reject_when_object_detected_in_region():
    from work.plate import is_clean
    dets = [{"label": "cup", "score": 0.6, "bbox": {"x": 500, "y": 700, "w": 200, "h": 300}}]
    region = (450, 560, 1285, 1830)
    assert is_clean(dets, region) is False


def test_accept_when_no_object_in_region():
    from work.plate import is_clean
    dets = [{"label": "mint leaf", "score": 0.5, "bbox": {"x": 40, "y": 1600, "w": 100, "h": 100}}]
    region = (450, 560, 1285, 1830)
    assert is_clean(dets, region) is True


def test_reject_multiple_bad_classes():
    from work.plate import is_clean
    dets = [
        {"label": "glass", "score": 0.7, "bbox": {"x": 100, "y": 100, "w": 50, "h": 50}},
        {"label": "bottle", "score": 0.8, "bbox": {"x": 500, "y": 700, "w": 200, "h": 300}},
    ]
    region = (450, 560, 1285, 1830)
    assert is_clean(dets, region) is False


def test_ignore_below_score_threshold():
    from work.plate import is_clean
    dets = [{"label": "cup", "score": 0.2, "bbox": {"x": 500, "y": 700, "w": 200, "h": 300}}]
    region = (450, 560, 1285, 1830)
    assert is_clean(dets, region) is True  # below default score_thr=0.35
