# i2e — Editable-Omnimatte Demo Design (single poster, zero-training)

**Date:** 2026-06-04
**Scope target:** Turn `IMG_9493.jpg` (风油精 × Häagen-Dazs collab poster, 1290×2202) into a
truly editable layered document. **Demo-first, zero new model training.** Train a custom model
only later, and only if this build's behind-hero plate proves insufficient.

---

## 1. Goal & success criteria

Produce an editor session where `IMG_9493.jpg` is decomposed into named layers over a clean
background plate, and these edits all work convincingly in a screen recording:

- **Recolor hero in place** — green ice-cream cup → pink / another flavor (hue shift on the layer).
- **Edit text / swap logo / relabel** — 风油精 → another brand; headline text editable.
- **Move / delete secondary objects** — bottle, mint, ice cubes; revealed area is clean.
- **Free-move the hero** — drag the cup elsewhere; its **smoke and shadow travel with it**
  (omnimatte), and the original location reveals an acceptably clean background.

**MVP is done when:** editor opens `IMG_9493` as ≥8 named layers + plate; all four edit classes
work; result looks convincing on screen.

**Why off-the-shelf is insufficient (established this session):** every fill/removal model tried
— LaMa, SD-1.5-Fill, FLUX.1-Fill, and SOTA **OmniEraser** (FLUX-based) — confabulates a
scene-plausible object instead of clean background in the large central hero region. Negative
prompts + true-CFG up to 4.0 + controlnet scale 0.95 do not fix it. Cause: training-objective +
domain gap + ill-posedness of a 26%-canvas central hole. Evidence: `work/poster/_omnieraser_analysis.png`.
This is why the hero-behind plate is crafted **once** for this poster rather than solved generally.

**Not a fit (checked):** CRAFTER / CraftEditor (arXiv 2605.30611) — same "raster→editable" framing
but a different domain (flat scientific figures → SVG vector) with **no** background inpainting,
amodal completion, or omnimatte. SVG cannot represent our photographic content (smoke, texture,
glossy reflections). We borrow only its multi-agent **critic loop** pattern (see §4, component ③).

---

## 2. Architecture overview

```
IMG_9493.jpg
   │
   ├─(have)──►  Layer inventory:  SAM3 modal masks + OCR text layers      [component ①]
   │
   ├─────────►  Omnimatte construction: per-object RGBA via removal-delta [component ②]
   │              → N RGBA layers (object + its own smoke/shadow)
   │              → raw plate (scene with everything removed)
   │
   ├─────────►  Plate crafting: CRAFTER-style critic loop on hero region  [component ③]
   │              → ONE clean "empty scene" plate PNG
   │
   ├─────────►  IR json: layers + transforms + plate references           [component ④a]
   │
   └─────────►  Editor: interactive move / recolor / relabel / delete     [component ④b]
                  → export flattened PNG
```

**Compute split:**
- Removal & generative fill (OmniEraser, FLUX) → **A800 docker box** (remote `29e8e3afb73f`,
  already provisioned: FLUX.1-dev + alimama controlnet + OmniEraser lora under
  `/home/lzy/AAAI_2026/i2e/`, isolated `pylibs` overlay, system env untouched).
- Omnimatte delta math, plate blend/compositing, editor → **local Mac** (no heavy GPU).

---

## 3. Components

### ① Layer inventory — HAVE IT
Reuse the existing `work/regen_sam3.py` output: SAM3 text-prompted modal masks + RapidOCR text
lines, assembled into the IR. Objects present: hero cup (tub `raster-1` + scoop `raster-2`),
perfume bottle, mint leaves, ice cubes, circular badges, brand logos, headline/body text lines.

### ② Omnimatte construction — NEW `work/omnimatte.py`
For each object, front-to-back in z-order (smaller area = nearer = removed first):

```
scene_without = OmniEraser.remove(scene_before, object_mask)     # remote call
delta         = scene_before − scene_without                     # per-pixel RGB difference
alpha         = normalize(|delta|), smoothed + thresholded       # soft smoke/shadow get partial alpha
layer.RGB     = original object pixels (from scene_before)
layer.alpha   = alpha                                            # object + its effects travel together
scene_before  = scene_without                                    # peel next, deeper object
```

After all objects are peeled, `scene_without_all` is the **raw plate**. The delta-alpha gives
"effects follow object" for free: a layer's smoke/shadow live in its own alpha channel, so moving
the layer moves them too.

**Known risk:** delta-alpha is noisy for diffuse smoke and for overlapping objects whose effects
intersect. Mitigation: Gaussian-smooth the alpha, low-threshold to kill speckle, and clamp alpha
to the object's dilated bbox so a layer cannot "carry" a distant unrelated change.

### ③ Plate crafting — NEW `work/plate.py` (the hard, one-off part)
The raw plate from ② is clean where removal was well-posed (bottle, mint, ice) but confabulates in
the hero region. So craft the hero region with a **CRAFTER-style critic loop**:

```
candidates = []
for seed in N_SEEDS:                       # diversity-driven parallel exploration
    cand = steered_fill(hero_hole, prompt="empty dark smoky studio, wet black marble,
                                            deep shadow, NO objects", seed)
critic(cand):                              # directive critic, not a scalar score
    - run SAM3 / a detector on the filled region
    - REJECT if any salient object/vessel/food is detected (the confabulation failure)
    - REJECT if blank/flat or seam-mismatched with surrounding real background
    - else ACCEPT, score by background-consistency with the poster's real dark regions
pick best ACCEPTED candidate (auto), with a manual final confirm
blend seam into surrounding real background (Poisson / feathered alpha)
→ ONE clean plate PNG for this poster
```

This is explicitly a **one-off artifact for `IMG_9493`** — it does not need to generalize. If no
seed survives the critic (hero region too large/ill-posed to ever fill cleanly), that is the
"must-train" signal → escalate to a poster-domain fine-tune (v2, out of scope here).

### ④a IR assembly — extend existing assemble
Write an IR json holding: plate reference; each layer's RGBA asset, z-order, and transform
(x, y, scale, rotation); text layers with content + font/box. Reuse `extractor/assemble.py`
conventions; add `transform` + `omnimatte` fields.

### ④b Editor extension — extend `editor/server.py` + `editor/editor.html`
- **Load:** plate as background + N RGBA omnimatte layers + text layers, each draggable with a
  transform handle and a z-order.
- **Move:** drag a layer over the plate; hero drag moves object+smoke+shadow together; original
  location reveals the plate.
- **Recolor:** HSV hue/saturation shift on a layer's RGB, alpha preserved (green→pink).
- **Relabel:** swap a text layer's content/font; swap a logo layer's raster.
- **Delete:** hide a layer → plate shows through.
- **Export:** flatten layers over plate → PNG.

Follow the editor's existing IR-load and asset-serving patterns; add per-layer transform + a
recolor control.

---

## 4. Data flow (end to end)

`IMG_9493 → SAM3+OCR (①) → omnimatte.py (② RGBA layers + raw plate) → plate.py (③ crafted plate)
→ IR.json (④a) → editor (④b interactive) → export PNG`

---

## 5. Out of scope (v2 / only when needed)
- Training a custom removal/de-render model (only if ③'s plate is unacceptable).
- Generalizing to arbitrary uploaded posters (this build targets `IMG_9493`).
- Physically-correct relighting of moved layers (we keep baked appearance).

## 6. Risks & mitigations
- **Behind-hero plate quality** (primary): best-of-N + critic + manual pick for one poster; escalate
  to fine-tune if no candidate survives.
- **Noisy omnimatte alpha** for diffuse smoke / overlapping effects: smooth + threshold + bbox clamp.
- **Remote round-trips** (each removal is a GPU call on a shared box): batch per-object removals in
  one loaded session (as the sweep already does); never preempt other users' GPU jobs.
- **Editor scope creep:** keep to the four edit classes; no general vector editing.
