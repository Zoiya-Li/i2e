"""Assemble work/poster/omnimatte.ir.json: plate + RGBA omnimatte layers (each with a
transform) + text layers. This IR is the contract the editor (follow-up plan) consumes."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "work/poster/omni_assets"


def build_ir(layers, plate, texts):
    out = {"canvas": {"w": layers["W"], "h": layers["H"]}, "plate": plate,
           "layers": [], "texts": texts}
    for L in layers["layers"]:
        out["layers"].append({
            "id": L["id"], "name": L["name"], "z": L["z"], "asset": L["asset"],
            "bbox": {"x": L["x"], "y": L["y"], "w": L["w"], "h": L["h"]},
            "transform": {"x": L["x"], "y": L["y"], "scale": 1.0, "rotation": 0.0}})
    return out


def main():
    layers = json.load(open(ASSETS / "layers.json"))
    ir0 = json.load(open(ROOT / "work/poster/poster.ir.json"))
    texts = []
    for el in ir0["elements"]:
        if el["type"] != "text":
            continue
        b = el["bbox"]
        ext = el.get("ext") or {}
        texts.append({"id": el["id"],
                      "content": ext.get("orig_content", ""),
                      "crop": ext.get("text_crop", ""),
                      "x": int(b["x"]), "y": int(b["y"]),
                      "w": int(b["w"]), "h": int(b["h"])})
    plate = str(ASSETS / "plate.png") if (ASSETS / "plate.png").exists() else str(ASSETS / "raw_plate.png")
    ir = build_ir(layers, plate, texts)
    p = ROOT / "work/poster/omnimatte.ir.json"
    p.write_text(json.dumps(ir, ensure_ascii=False, indent=2))
    print(f"wrote {p}: {len(ir['layers'])} layers, {len(ir['texts'])} texts, plate={Path(plate).name}")


if __name__ == "__main__":
    main()
