"""Second hybrid specialist: segmentation. Turns photographic foreground
elements (product shots) into real RGBA cutout layers + masks. Division of
labor — VLM localizes WHICH box is a product; the segmenter cuts the precise
pixels. rembg (onnxruntime, light) for the MVP; a promptable SAM-2 slot is left
for later (GPU + box-prompted "cut this specific thing").
"""
