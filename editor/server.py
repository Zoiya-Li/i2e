"""Node ③ backend: serve the editor, the IR, and the original image; on save,
diff the edited IR against Node ②'s prediction and capture corrections (Node ⑤).

    python -m editor.server <predicted.ir.json> [--out edited.ir.json] [--port 8765]

The original (predicted) IR loaded at startup is the diff baseline — corrections
are always (model predicted -> user shipped), not (last save -> this save).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from capture.corrections import append_corrections, capture_diff
from extractor.assemble import validate_ir
from render.export import render

_HTML = (Path(__file__).parent / "editor.html").read_text()
_CT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
       ".webp": "image/webp", ".gif": "image/gif"}


def apply_save(original_ir: dict, edited_ir: dict, out_path: str,
               session_id: str | None = None) -> dict:
    """Testable core of Node ③→⑤: capture corrections, validate, persist.
    Returns {corrections, count, out}. Pure IR->IR (no image/network needed)."""
    corrections = capture_diff(original_ir, edited_ir, session_id=session_id)
    shipped = copy.deepcopy(edited_ir)
    append_corrections(shipped, corrections)
    validate_ir(shipped)  # never persist an invalid IR
    Path(out_path).write_text(json.dumps(shipped, ensure_ascii=False, indent=2))
    return {"corrections": corrections, "count": len(corrections), "out": out_path}


def _allowed_assets(ir: dict, image_path: str) -> set[str]:
    """Exact set of asset paths the IR references — only these are servable."""
    allowed = {os.path.abspath(image_path)}
    for el in ir["elements"]:
        for p in [(el.get("background") or {}).get("asset_ref"),
                  (el.get("ext") or {}).get("cutout"),
                  (el.get("ext") or {}).get("text_crop"),
                  (el.get("raster") or {}).get("asset_ref"),
                  (el.get("logo") or {}).get("raster_ref")]:
            if p:
                allowed.add(os.path.abspath(p))
    return allowed


def _make_handler(original_ir: dict, image_path: str, out_path: str, seg_dir: str):
    allowed_assets = _allowed_assets(original_ir, image_path)
    seg_abs = os.path.abspath(seg_dir)
    Path(seg_abs).mkdir(parents=True, exist_ok=True)
    seg_uid = [0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            data = body if isinstance(body, (bytes, bytearray)) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, _HTML, "text/html; charset=utf-8")
            elif self.path == "/api/ir":
                self._send(200, json.dumps(original_ir, ensure_ascii=False), "application/json")
            elif self.path == "/image":
                p = Path(image_path)
                if not p.exists():
                    self._send(404, b"image not found", "text/plain"); return
                self._send(200, p.read_bytes(), _CT.get(p.suffix.lower(), "application/octet-stream"))
            elif self.path.startswith("/asset"):
                q = parse_qs(urlparse(self.path).query)
                ap = os.path.abspath(unquote(q.get("p", [""])[0]))
                if (ap in allowed_assets or ap.startswith(seg_abs + os.sep)) and Path(ap).exists():
                    self._send(200, Path(ap).read_bytes(), _CT.get(Path(ap).suffix.lower(), "application/octet-stream"))
                else:
                    self._send(404, b"asset not allowed", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path not in ("/api/save", "/api/export", "/api/segment", "/api/commit", "/api/complete", "/api/bulk_segment", "/api/find_missing"):
                self._send(404, b"not found", "text/plain"); return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/api/save":
                    result = apply_save(original_ir, body, out_path, session_id="editor")
                    self._send(200, json.dumps({"count": result["count"],
                                                "corrections": result["corrections"]},
                                               ensure_ascii=False), "application/json")
                elif self.path == "/api/export":  # render edited IR to a shipped PNG (Node ④)
                    self._send(200, render(body, original_ir, image_path), "image/png")
                elif self.path == "/api/segment":  # user-driven SAM point/box prompt
                    from segment.interactive import segment
                    seg_uid[0] += 1
                    res = segment(image_path, body.get("mode", "point"),
                                  points=body.get("points"), labels=body.get("labels"),
                                  box=body.get("box"), clip_bbox=body.get("clip_bbox"),
                                  out_dir=seg_abs, uid=f"seg-{seg_uid[0]}")
                    self._send(200 if res.get("ok") else 400, json.dumps(res), "application/json")
                elif self.path == "/api/commit":  # remove the split part from its parent's surface
                    from segment.interactive import inpaint_parent
                    seg_uid[0] += 1
                    out = os.path.join(seg_abs, f"parent-{seg_uid[0]}.png")
                    inpaint_parent(body["parent_cutout"], body["parent_bbox"],
                                   body["mask_ref"], out)
                    self._send(200, json.dumps({"ok": True, "parent_cutout": out}), "application/json")
                elif self.path == "/api/complete":  # amodal completion of an occluded layer
                    from segment.interactive import complete_occlusions
                    seg_uid[0] += 1
                    res = complete_occlusions(image_path, body["target_cutout"], body["target_bbox"],
                                              body.get("occluders", []), seg_abs, f"amodal-{seg_uid[0]}")
                    self._send(200 if res.get("ok") else 400, json.dumps(res), "application/json")
                elif self.path == "/api/bulk_segment":  # dense SAM inside a user-lassoed region
                    from segment.interactive import bulk_segment
                    seg_uid[0] += 1
                    cands = bulk_segment(image_path, body["box"], seg_abs, f"bulk-{seg_uid[0]}")
                    self._send(200, json.dumps({"ok": True, "candidates": cands}), "application/json")
                else:  # /api/find_missing — VLM scans bg plate, suggests un-extracted elements
                    from label.vlm import find_missing
                    # use the bg plate as the "what's left" picture
                    bg = None
                    for el in original_ir["elements"]:
                        if el.get("type") == "background":
                            bg = (el.get("background") or {}).get("asset_ref"); break
                    items = find_missing(bg or image_path) if bg else []
                    self._send(200, json.dumps({"ok": True, "items": items}), "application/json")
            except Exception as e:
                self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")
    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="i2e editor (Node ③)")
    ap.add_argument("ir", help="predicted IR json (from extractor)")
    ap.add_argument("--out", help="where to write the edited IR (default: <ir>.edited.json)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    original_ir = json.loads(Path(args.ir).read_text())
    image_path = original_ir["source"]["original_image_ref"]
    # keep the edited filename ending in ".ir.json" so bench/flywheel.py (globs *.ir.json)
    # counts it; was ".ir.edited.json" which the metric silently skipped.
    if args.out:
        out_path = args.out
    elif args.ir.endswith(".ir.json"):
        out_path = args.ir[: -len(".ir.json")] + ".edited.ir.json"
    else:
        out_path = str(Path(args.ir).with_suffix(".edited.json"))
    seg_dir = str(Path(out_path).parent / "_seg")

    handler = _make_handler(original_ir, image_path, out_path, seg_dir)
    # find a free port starting at --port, so we never collide with another server
    import socket
    port = None
    for cand in range(args.port, args.port + 40):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", cand)); s.close(); port = cand; break
        except OSError:
            s.close()
    if port is None:
        print(f"no free port near {args.port}", file=sys.stderr); return 1
    if port != args.port:
        print(f"(port {args.port} busy — using {port})")
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"i2e editor → http://127.0.0.1:{port}   (image: {image_path}, saves to: {out_path})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
