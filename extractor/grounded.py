"""Open-vocabulary object detection with GroundingDINO.

Given a list of text queries (element names from the VLM), returns trustworthy
bounding boxes for each. This is the missing layer between SAM (great masks,
no semantics) and a VLM (great semantics, no spatial precision).

Usage:
    from extractor.grounded import detect
    dets = detect("IMG.jpg", ["ice cream cup", "mint leaf", "circular badge"])
    # dets: list of {label, score, bbox: {x,y,w,h}}
"""
from __future__ import annotations
import os
import threading

_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None
_DEVICE = None
_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _load():
    """Lazy-load model + processor; MPS if available, else CPU."""
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _DEVICE
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _PROCESSOR, _DEVICE
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        os.environ.setdefault("HF_HOME", os.path.abspath("work/hf_cache"))
        _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        _PROCESSOR = AutoProcessor.from_pretrained(_MODEL_ID)
        _MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID).to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _PROCESSOR, _DEVICE


def detect(image_path: str, queries, box_threshold: float = 0.30,
           text_threshold: float = 0.25) -> list:
    """Detect each query in the image. Returns [{label, score, bbox{x,y,w,h}}]."""
    import torch
    from PIL import Image
    if not queries:
        return []
    model, processor, device = _load()
    image = Image.open(image_path).convert("RGB")
    # GroundingDINO format: lowercase, dot-separated queries ending with "."
    text = ". ".join(q.lower().strip(".") for q in queries) + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # post_process_grounded_object_detection: signature varies by transformers version
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
    except TypeError:
        # newer transformers uses `threshold` + `text_threshold`
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
    out = []
    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels") or results.get("text_labels") or []
    for i in range(len(boxes)):
        x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        out.append({
            "label": str(labels[i]) if i < len(labels) else "",
            "score": float(scores[i]),
            "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
        })
    out.sort(key=lambda d: -d["score"])
    return out
