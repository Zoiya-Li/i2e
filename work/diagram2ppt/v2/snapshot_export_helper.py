"""Clean subprocess helper for PowerPoint PDF export."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from work.diagram2ppt.v2.snapshot import _export_pdf_local


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: snapshot_export_helper PPTX PDF [ATTEMPTS]", file=sys.stderr)
        return 2
    pptx = Path(sys.argv[1]).resolve()
    pdf = Path(sys.argv[2]).resolve()
    attempts = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.environ.pop("MPLCONFIGDIR", None)
    produced = _export_pdf_local_to_path(pptx, pdf, attempts)
    print(produced)
    return 0


def _export_pdf_local_to_path(pptx: Path, pdf: Path, attempts: int) -> Path:
    # Reuse the local exporter by temporarily choosing the requested output name.
    # This helper is already isolated from matplotlib-heavy build code.
    from work.diagram2ppt.v2 import snapshot as snap

    original = snap._fresh_pdf_path
    snap._fresh_pdf_path = lambda _pptx, _attempt: pdf
    try:
        return _export_pdf_local(pptx, attempts)
    finally:
        snap._fresh_pdf_path = original


if __name__ == "__main__":
    raise SystemExit(main())
