"""Local model loaders for v3.

Each loader downloads the model to the given cache_dir (reusing existing HF/
MS caches when possible) and returns the objects needed by the wrapper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_grounding_dino(cache_dir: str, device: str) -> tuple[Any, Any]:
    """Load IDEA-Research/grounding-dino-tiny."""
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers required for grounding-dino") from exc

    model_id = "IDEA-Research/grounding-dino-tiny"
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, **kwargs)
    model = model.to(device).eval()
    return model, processor


def load_sam3(cache_dir: str, device: str) -> Any:
    """Load SAM3 from the local ms_cache or external/sam3."""
    import torch

    # Prefer the explicit local paths already present in the project.
    candidates = [
        Path(cache_dir) / "facebook/sam3/sam3.pt",
        Path("external/sam3/sam3.pt"),
        Path("work/ms_cache/facebook/sam3/sam3.pt"),
    ]
    checkpoint = None
    for c in candidates:
        if c.exists():
            checkpoint = str(c)
            break
    if checkpoint is None:
        raise FileNotFoundError(
            "sam3.pt not found in local caches; expected one of: "
            + ", ".join(str(c) for c in candidates)
        )

    try:
        # SAM3 uses the segment_anything package or ultralytics.
        from ultralytics import SAM
        model = SAM(checkpoint)
        if hasattr(model, "to"):
            model = model.to(device)
        return model
    except Exception as exc:
        raise RuntimeError(
            "Could not load SAM3. Install with: pip install ultralytics"
        ) from exc


def load_pix2tex(cache_dir: str, device: str) -> tuple[Any, Any, Any]:
    """Load local formula OCR model (image → LaTeX).

    Primary: breezedeus/pix2text-mfr (ONNX TrOCR, fast on CPU/MPS).
    Fallback: vikp/texify if the ONNX model is unavailable.
    """
    import os
    from pathlib import Path

    # Models were downloaded from huggingface.co directly; the configured
    # HF_ENDPOINT mirror may not have them, so bypass it for local loading.
    old_endpoint = os.environ.pop("HF_ENDPOINT", None)
    try:
        # Prefer the local ONNX model that was just downloaded.
        local_dir = Path(__file__).parent / "pix2text-mfr"
        if local_dir.exists():
            return load_pix2text_mfr(str(local_dir), device)

        # Optional texify fallback (PyTorch, more finicky on MPS).
        texify_dir = Path(__file__).parent / "texify"
        if texify_dir.exists():
            return load_texify(str(texify_dir), device)
        raise FileNotFoundError(
            "No local formula model found. Expected one of: "
            f"{local_dir}, {texify_dir}"
        )
    finally:
        if old_endpoint is not None:
            os.environ["HF_ENDPOINT"] = old_endpoint


def load_texify(local_dir: str, device: str) -> tuple[Any, Any, Any]:
    """Load vikp/texify (PyTorch VisionEncoderDecoder)."""
    try:
        from transformers import AutoProcessor, AutoTokenizer, VisionEncoderDecoderModel
    except ImportError as exc:
        raise RuntimeError("transformers required for texify") from exc

    kwargs = {"local_files_only": True}
    model = VisionEncoderDecoderModel.from_pretrained(local_dir, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, **kwargs)
    processor = AutoProcessor.from_pretrained(local_dir, **kwargs)
    model = model.to(device).eval()
    return model, tokenizer, processor


def load_pix2text_mfr(local_dir: str, _device: str) -> tuple[Any, Any, Any]:
    """Load breezedeus/pix2text-mfr (ONNX TrOCR).

    Returns a tuple of (encoder_session, decoder_session, processor).  The
    device argument is accepted for API compatibility but ONNX Runtime uses
    its own provider list.
    """
    import os

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime required for pix2text-mfr") from exc

    try:
        from transformers import TrOCRProcessor
    except ImportError as exc:
        raise RuntimeError("transformers required for pix2text-mfr processor") from exc

    local_path = Path(local_dir)
    encoder_path = local_path / "encoder_model.onnx"
    decoder_path = local_path / "decoder_model.onnx"
    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"pix2text-mfr ONNX files missing in {local_dir}"
        )

    # Use CoreML/MPS on Apple silicon when available; fall back to CPU.
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    encoder = ort.InferenceSession(
        str(encoder_path), sess_options=sess_options, providers=providers
    )
    decoder = ort.InferenceSession(
        str(decoder_path), sess_options=sess_options, providers=providers
    )
    processor = TrOCRProcessor.from_pretrained(local_dir)
    return encoder, decoder, processor
