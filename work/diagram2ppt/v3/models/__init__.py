"""Local model registry for diagram2ppt v3.

Models are downloaded to work/diagram2ppt/v3/models/cache (or reuse existing
work/hf_cache / work/ms_cache). Agents request a model by name; the registry
returns a callable wrapper.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .loader import load_grounding_dino, load_pix2tex, load_sam3

CACHE_ROOT = Path(__file__).parent / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Allow reuse of the project's existing model caches.
_HF_CACHE = Path("work/hf_cache/hub")
_MS_CACHE = Path("work/ms_cache")


class LocalModel:
    """Wrapper around a loaded local model."""

    def __init__(self, name: str, fn: Callable, device: str = "cpu") -> None:
        self.name = name
        self.fn = fn
        self.device = device

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def get_local_model(name: str, cache_dir: str | None = None) -> LocalModel:
    """Load or return a cached local model.

    Supported names:
      - "grounding-dino-tiny"  → object detection with text prompts
      - "sam3"                 → image segmentation
      - "pix2tex"              → formula image → LaTeX
    """
    device = _auto_device()
    if name == "grounding-dino-tiny":
        model, processor = load_grounding_dino(
            cache_dir=cache_dir or str(_HF_CACHE), device=device)
        return LocalModel(name, _wrap_grounding_dino(model, processor, device), device)
    if name == "sam3":
        model = load_sam3(cache_dir=cache_dir or str(_MS_CACHE), device=device)
        return LocalModel(name, _wrap_sam3(model, device), device)
    if name == "pix2tex":
        model, tokenizer, processor = load_pix2tex(
            cache_dir=cache_dir or str(CACHE_ROOT), device=device)
        return LocalModel(name, _wrap_pix2tex(model, tokenizer, processor, device), device)
    raise ValueError(f"unsupported local model: {name}")


def _auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _wrap_grounding_dino(model: Any, processor: Any, device: str) -> Callable:
    def detect(image, text: str, threshold: float = 0.3, nms_threshold: float = 0.5):
        import torch
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results.get("text_labels") or results.get("labels") or [""] * len(boxes)
        # simple NMS
        keep = _nms(boxes, scores, nms_threshold)
        return [
            {"bbox": [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]],
             "score": scores[i],
             "label": labels[i]}
            for i in keep
        ]
    return detect


def _wrap_sam3(model: Any, device: str) -> Callable:
    def segment(image, boxes: list[list[float]] | None = None):
        import numpy as np
        import torch
        model.set_image(np.array(image))
        if boxes:
            input_boxes = torch.tensor(boxes, device=device)
            masks, _, _ = model.predict_torch(
                point_coords=None, point_labels=None,
                boxes=input_boxes, multimask_output=False)
        else:
            # automatic mask generation requires SAM3 AMG wrapper; skip for now
            masks = None
        return masks
    return segment


def _wrap_pix2tex(model: Any, tokenizer_or_decoder: Any, processor: Any,
                  device: str) -> Callable:
    """Dispatch between ONNX pix2text-mfr and PyTorch texify wrappers."""
    # The loader returns either (encoder_session, decoder_session, processor)
    # for the ONNX model, or (pytorch_model, tokenizer, processor) for texify.
    if hasattr(model, "run"):
        return _wrap_pix2text_mfr(model, tokenizer_or_decoder, processor)
    return _wrap_texify(model, tokenizer_or_decoder, processor, device)


def _wrap_texify(model: Any, tokenizer: Any, processor: Any, device: str) -> Callable:
    def predict(image):
        import torch
        from torchvision import transforms

        # Manual preprocessing matching texify's VariableDonutImageProcessor:
        # resize to 420x420, rescale to [0,1], normalize with ImageNet stats.
        t = transforms.Compose([
            transforms.Resize((420, 420)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = t(image).unsqueeze(0).to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=512)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predict


def _wrap_pix2text_mfr(encoder_session: Any, decoder_session: Any,
                       processor: Any) -> Callable:
    """ONNX pix2text-mfr wrapper with beam search decoding."""
    tokenizer = processor.tokenizer
    decoder_start_token_id = 2
    eos_token_id = 2
    vocab_size = 1200
    beam_width = 3
    max_new_tokens = 256

    def _decoder_logits(input_ids: np.ndarray, encoder_hidden_states: np.ndarray):
        return decoder_session.run(
            None,
            {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states},
        )[0]

    def predict(image):
        pixel_values = processor.image_processor(
            image, return_tensors="np"
        )["pixel_values"].astype(np.float32)
        encoder_hidden_states = encoder_session.run(
            None, {"pixel_values": pixel_values}
        )[0]

        # Beam search: each item is (score, tokens).
        beams = [(0.0, [decoder_start_token_id])]
        complete: list[tuple[float, list[int]]] = []

        for _ in range(max_new_tokens):
            candidates: list[tuple[float, list[int]]] = []
            for score, tokens in beams:
                # A beam is complete once it has generated at least one real
                # token and ends with EOS.  The start token itself is EOS, so
                # we require length > 1.
                if len(tokens) > 1 and tokens[-1] == eos_token_id:
                    complete.append((score, tokens))
                    continue
                input_ids = np.array([tokens], dtype=np.int64)
                logits = _decoder_logits(input_ids, encoder_hidden_states)
                log_probs = _log_softmax(logits[0, -1])
                # Keep top-k per beam.
                topk = min(beam_width, vocab_size)
                best_indices = np.argpartition(log_probs, -topk)[-topk:]
                best_indices = best_indices[np.argsort(-log_probs[best_indices])]
                for idx in best_indices:
                    idx = int(idx)
                    candidates.append(
                        (score + log_probs[idx], tokens + [idx])
                    )
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:beam_width]
            # Early stop if all beams are complete.
            if all(len(b) > 1 and b[-1] == eos_token_id for _, b in beams):
                break

        if complete:
            complete.sort(key=lambda x: -x[0])
            best_tokens = complete[0][1]
        elif beams:
            best_tokens = beams[0][1]
        else:
            best_tokens = [decoder_start_token_id]

        latex = tokenizer.decode(best_tokens, skip_special_tokens=True)
        return _clean_formula_latex(latex)

    return predict


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return np.log(e_x / np.sum(e_x))


def _clean_formula_latex(latex: str) -> str:
    """Trim repetitive padding and structural wrappers produced on tight crops."""
    latex = latex.strip()
    # Remove leading/trailing \qquad and \quad spacing fillers.
    latex = re.sub(r"^(\\qquad|\\quad)\s*", "", latex)
    latex = re.sub(r"\s*(\\qquad|\\quad)$", "", latex)
    # Collapse repeated spacing commands.
    latex = re.sub(r"(\\qquad\s*)+", r"\\qquad ", latex)
    latex = re.sub(r"(\\quad\s*)+", r"\\quad ", latex)
    # The box/rectangle around the formula is usually a separate shape element
    # in the IR; keeping \fbox/\boxed would double-render it.
    for wrapper in ("fbox", "boxed"):
        latex = _strip_outer_wrapper(latex, wrapper)
    return latex.strip()


def _strip_outer_wrapper(latex: str, cmd: str) -> str:
    """Remove a single outer LaTeX wrapper command, e.g. \\fbox{...}."""
    prefix = "\\" + cmd
    latex = latex.strip()
    if not latex.startswith(prefix):
        return latex
    # Skip the command and optional whitespace, then expect '{'.
    idx = len(prefix)
    while idx < len(latex) and latex[idx].isspace():
        idx += 1
    if idx >= len(latex) or latex[idx] != "{":
        return latex
    # Find the matching closing brace, respecting nesting.
    depth = 0
    start = idx  # index of the opening '{'
    for i in range(start, len(latex)):
        if latex[i] == "{":
            depth += 1
        elif latex[i] == "}":
            depth -= 1
            if depth == 0:
                return latex[start + 1:i].strip()
    return latex


def _nms(boxes: list[list[float]], scores: list[float],
         threshold: float = 0.5) -> list[int]:
    """Greedy IoU-based NMS."""
    if not boxes:
        return []
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while indices:
        current = indices.pop(0)
        keep.append(current)
        cx0, cy0, cx1, cy1 = boxes[current]
        c_area = max(0, cx1 - cx0) * max(0, cy1 - cy0)
        indices = [
            i for i in indices
            if _iou(boxes[i], boxes[current]) < threshold
        ]
    return keep


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = max(0, ax1 - ax0) * max(0, ay1 - ay0) + max(0, bx1 - bx0) * max(0, by1 - by0) - inter
    return inter / union if union else 0.0
