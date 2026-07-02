"""Node ④ (lightweight) — render an edited IR back to a shipped flat image.

MVP scope: handles TEXT edits (copy / localization / color / reposition) by
covering the original baked-in text with a sampled background color and
re-drawing the new text. Photographic-element moves and faithful font matching
wait for the real asset pipeline (SAM cutout + inpaint). Good enough to give the
high-volume marketing wedge a usable asset for the most common edit — which is
the half of the flywheel that makes users come back.
"""
