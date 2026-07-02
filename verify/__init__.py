"""Real needs_review, computed from evidence instead of the model's (useless)
self-reported confidence. Render-independent signals: geometry sanity + OCR
cross-check (does the recognized text actually sit where/what the IR claims).
Each flagged element records WHY in ext.review so the editor can guide the user.
"""
