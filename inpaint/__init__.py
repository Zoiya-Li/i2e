"""Third specialist: inpainting / background reconstruction — the long-flagged
biggest lossy point. Removes foreground elements and fills the hole, producing a
clean background plate. With cutouts + a clean background, the IR becomes a true
layered document: elements can move/be removed and the gap fills instead of
ghosting. OpenCV (cv2.inpaint, light, no model) for the MVP; a LaMa/IOPaint slot
is left for higher-quality photographic fills later.
"""
