"""True PPTX renderer for v3.

Renders a PPTX file through the real PowerPoint application (AppleScript → PDF →
PNG on macOS) so the Verifier sees the same pixels a user would see.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from work.diagram2ppt.v2.snapshot import compare as v2_compare
from work.diagram2ppt.v2.snapshot import snapshot as v2_snapshot


def render(pptx_path: str, out_png: str | None = None, dpi: int = 130) -> str:
    """Render a PPTX to PNG via real PowerPoint.

    Args:
        pptx_path: input .pptx path.
        out_png: optional output PNG path.
        dpi: render resolution.

    Returns:
        Path to the rendered PNG.
    """
    pptx = Path(pptx_path)
    if not pptx.exists():
        raise FileNotFoundError(pptx_path)
    out = out_png or str(pptx.with_suffix(".true.png"))
    return v2_snapshot(str(pptx), out, dpi=dpi)


def compare(pptx_path: str, original_path: str, out_png: str | None = None,
            dpi: int = 130) -> str:
    """Render PPTX and stack it under the original image for visual inspection.

    Args:
        pptx_path: input .pptx path.
        original_path: reference image path.
        out_png: optional output comparison PNG path.
        dpi: render resolution.

    Returns:
        Path to the comparison PNG.
    """
    pptx = Path(pptx_path)
    original = Path(original_path)
    if not pptx.exists():
        raise FileNotFoundError(pptx_path)
    if not original.exists():
        raise FileNotFoundError(original_path)
    out = out_png or str(pptx.with_suffix(".compare.png"))
    return v2_compare(str(pptx), str(original), out, dpi=dpi)


def render_isolated(pptx_path: str, out_png: str | None = None, dpi: int = 130) -> str:
    """Render through PowerPoint in a fresh Python process.

    Building PPTX files imports plotting/rendering libraries and can leave the
    process in a state where AppleScript/PDF export is flaky.  The planner
    should judge candidates from real PowerPoint pixels, so rendering is run in
    a clean subprocess that only imports the snapshot stack.
    """
    pptx = Path(pptx_path)
    if not pptx.exists():
        raise FileNotFoundError(pptx_path)
    out = Path(out_png or str(pptx.with_suffix(".true.png")))
    _run_isolated(["render", str(pptx), str(out), "--dpi", str(int(dpi))])
    if not out.exists():
        raise RuntimeError(f"isolated render did not create {out}")
    return str(out)


def compare_isolated(
    pptx_path: str,
    original_path: str,
    out_png: str | None = None,
    dpi: int = 130,
) -> str:
    """Create a true PowerPoint comparison image in a fresh Python process."""
    pptx = Path(pptx_path)
    original = Path(original_path)
    if not pptx.exists():
        raise FileNotFoundError(pptx_path)
    if not original.exists():
        raise FileNotFoundError(original_path)
    out = Path(out_png or str(pptx.with_suffix(".compare.png")))
    _run_isolated([
        "compare",
        str(pptx),
        str(original),
        str(out),
        "--dpi",
        str(int(dpi)),
    ])
    if not out.exists():
        raise RuntimeError(f"isolated compare did not create {out}")
    return str(out)


def _run_isolated(args: list[str]) -> None:
    env = os.environ.copy()
    env.pop("MPLCONFIGDIR", None)
    env.pop("I2E_DISABLE_POWERPOINT_RENDER", None)
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root) if not pythonpath else f"{repo_root}{os.pathsep}{pythonpath}"
    )
    timeout = _render_timeout_sec()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "work.diagram2ppt.v3.render_compare", *args],
            cwd=str(repo_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"isolated PowerPoint render timed out after {timeout}s"
        ) from exc
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"isolated PowerPoint render failed: {msg}")


def _render_timeout_sec() -> int:
    raw = os.environ.get("I2E_POWERPOINT_RENDER_TIMEOUT", "25")
    try:
        return max(3, min(300, int(raw)))
    except ValueError:
        return 25


def is_available() -> bool:
    """Return True if real PowerPoint rendering is available on this machine."""
    if os.environ.get("I2E_DISABLE_POWERPOINT_RENDER", "").lower() in {"1", "true", "yes"}:
        return False
    return shutil.which("osascript") is not None
