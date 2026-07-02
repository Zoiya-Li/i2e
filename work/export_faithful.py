"""Export the FAITHFUL layered document as an SVG a designer opens in Illustrator/Figma.

Structure mirrors how the poster is built:
  <g id="scene">          the photographic scene (backdrop image, products/scene kept)
  <g id="overlay-graphics">  logo / badge / icon — original-pixel <image> layers
  <g id="overlay-text">      each text line — original-pixel <image> layer, carrying its content,
                             colour and a font *hint* as data-attrs so a designer can retype it
Every element is original pixels (faithful, zero degradation); editing = hide the layer and
retype in your own font. Writes work/poster/faithful/poster_faithful.svg.
"""
import json, sys, base64
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
F = ROOT / "work/poster/faithful"


def uri(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def img(L):
    return (f'<image id="{L["id"]}" href="{uri(L["asset"])}" x="{L["x"]}" y="{L["y"]}" '
            f'width="{L["w"]}" height="{L["h"]}" data-kind="{L["kind"]}" '
            f'data-content="{esc(L.get("content",""))}" data-color="{L.get("color","")}" '
            f'data-font-hint="{esc(L.get("font_hint",""))}"/>')


def main():
    ir = json.load(open(F / "faithful.ir.json"))
    W, H = ir["canvas"]["w"], ir["canvas"]["h"]
    layers = ir["layers"]
    graphics = [L for L in layers if L["kind"] == "graphic"]
    texts = [L for L in layers if L["kind"] == "text"]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    parts.append(f'<g id="scene"><image id="backdrop" href="{uri(ir["backdrop"])}" '
                 f'x="0" y="0" width="{W}" height="{H}"/></g>')
    parts.append('<g id="overlay-graphics">')
    parts += [img(L) for L in graphics]
    parts.append('</g>')
    parts.append('<g id="overlay-text">')
    parts += [img(L) for L in texts]
    parts.append('</g></svg>')

    out = F / "poster_faithful.svg"; out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  scene(1) + overlay-graphics({len(graphics)}) + overlay-text({len(texts)}) = {1+len(layers)} layers")
    import os
    print(f"  {os.path.getsize(out)//1024} KB; open in Illustrator/Figma/browser")


if __name__ == "__main__":
    main()
