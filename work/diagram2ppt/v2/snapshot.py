"""Ground-truth visual check: render a PPTX through REAL PowerPoint.

The PIL proxy and QuickLook both lie (autofit, OMML, custGeom differ);
Microsoft PowerPoint itself is installed, so use it: AppleScript export to
PDF (its PNG export is broken/sandboxed) → PyMuPDF → PNG you can LOOK at.

    python -m work.diagram2ppt.v2.snapshot deck.pptx [-o out.png] [--dpi 130]

Rule (memory: feedback-render-and-look): after building any deck, run this
and look at the image BEFORE claiming anything about visual quality.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

SANDBOX_DOCS = Path.home() / ("Library/Containers/com.microsoft.Powerpoint"
                              "/Data/Documents")

APPLESCRIPT = '''
tell application "Microsoft PowerPoint"
    activate
    close every presentation saving no
    set pptAlias to POSIX file "{pptx}"
    open pptAlias
    set pres to active presentation
    set pdfAlias to POSIX file "{pdf}"
    save pres in pdfAlias as save as PDF
    close pres saving no
end tell
'''


def snapshot(pptx_path: str, out_png: str | None = None, dpi: int = 130) -> str:
    pptx = Path(pptx_path).resolve()
    out = Path(out_png) if out_png else pptx.with_suffix(".true.png")
    pdf = _export_pdf_with_powerpoint(pptx)

    import fitz

    doc = fitz.open(str(pdf))
    doc[0].get_pixmap(dpi=dpi).save(str(out))
    return str(out)


def _export_pdf_with_powerpoint(pptx: Path, attempts: int = 3) -> Path:
    """Export through PowerPoint, tolerating transient AppleScript failures."""
    SANDBOX_DOCS.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("I2E_PPT_EXPORT_HELPER") and _needs_clean_export_process():
        return _export_pdf_in_clean_process(pptx, attempts)
    return _export_pdf_local(pptx, attempts)


def _export_pdf_local(pptx: Path, attempts: int = 3) -> Path:
    last_detail = ""
    last_pdf: Path | None = None
    for attempt in range(attempts):
        pdf = _fresh_pdf_path(pptx, attempt)
        last_pdf = pdf
        started_at = time.time() - 0.25
        proc = subprocess.run(
            ["osascript", "-e", APPLESCRIPT.format(pptx=pptx, pdf=pdf)],
            check=False,
            capture_output=True,
            timeout=120,
            text=True,
            env=_osascript_env(),
        )
        for _ in range(20):           # PowerPoint writes asynchronously
            if pdf.exists() and pdf.stat().st_size > 0 and pdf.stat().st_mtime >= started_at:
                return pdf
            time.sleep(0.5)
        last_detail = (proc.stderr or proc.stdout or "").strip()
        _quit_powerpoint()
        time.sleep(0.6 + attempt * 0.4)

    if last_detail:
        raise RuntimeError(f"PowerPoint export failed: {last_detail}")
    raise RuntimeError(f"PowerPoint did not produce a fresh {last_pdf}")


def _export_pdf_in_clean_process(pptx: Path, attempts: int = 3) -> Path:
    pdf = _fresh_pdf_path(pptx, 0)
    env = _minimal_helper_env()
    env["I2E_PPT_EXPORT_HELPER"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "work.diagram2ppt.v2.snapshot_export_helper",
            str(pptx),
            str(pdf),
            str(attempts),
        ],
        check=False,
        capture_output=True,
        timeout=180,
        text=True,
        env=env,
    )
    if pdf.exists() and pdf.stat().st_size > 0:
        return pdf
    detail = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(f"clean PowerPoint export helper failed: {detail}")


def _fresh_pdf_path(pptx: Path, attempt: int) -> Path:
    unique = hashlib.sha1(
        f"{pptx}:{time.time_ns()}:{os.getpid()}:{attempt}".encode()
    ).hexdigest()[:10]
    return SANDBOX_DOCS / f"_i2e_ppt_render_{unique}.pdf"


def _needs_clean_export_process() -> bool:
    return "MPLCONFIGDIR" in os.environ or "matplotlib" in sys.modules


def _quit_powerpoint() -> None:
    subprocess.run(
        ["osascript", "-e", 'tell application "Microsoft PowerPoint" to quit saving no'],
        check=False,
        capture_output=True,
        timeout=30,
        text=True,
        env=_osascript_env(),
    )


def _osascript_env() -> dict[str, str]:
    env = os.environ.copy()
    # Matplotlib/PyMuPDF test harnesses often set MPLCONFIGDIR.  Passing it to
    # osascript can make PowerPoint's scripting terminology lookup fail with
    # "Expected class name but found identifier" on macOS.
    env.pop("MPLCONFIGDIR", None)
    return env


def _minimal_helper_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (
        "HOME", "USER", "LOGNAME", "TMPDIR", "PATH", "LANG", "LC_ALL",
        "SHELL", "__CF_USER_TEXT_ENCODING", "SECURITYSESSIONID",
        "SSH_AUTH_SOCK", "XPC_FLAGS", "XPC_SERVICE_NAME", "COMMAND_MODE",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    return env


def compare(pptx_path: str, original_path: str, out_png: str | None = None,
            dpi: int = 130) -> str:
    """Snapshot + stack under the original. ALWAYS look at this, not the
    snapshot alone — 'looks fine' means nothing without the reference."""
    from PIL import Image

    snap = snapshot(pptx_path, None, dpi)
    o = Image.open(original_path).convert("RGB")
    r = Image.open(snap).convert("RGB")
    r = r.resize((o.width, int(r.height * o.width / r.width)))
    side = Image.new("RGB", (o.width, o.height + r.height + 8), "red")
    side.paste(o, (0, 0))
    side.paste(r, (0, o.height + 8))
    out = out_png or str(Path(pptx_path).with_suffix(".compare.png"))
    side.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pptx")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--compare", metavar="ORIGINAL",
                    help="also stack the render under this original image")
    a = ap.parse_args()
    if a.compare:
        print(compare(a.pptx, a.compare, a.out, a.dpi))
    else:
        print(snapshot(a.pptx, a.out, a.dpi))
