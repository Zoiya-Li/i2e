"""Node ② — de-layer a flat image into a schema-valid IR.

Provider-agnostic by design (the "be Switzerland" principle): the VLM used
internally is swappable. `mock` needs no API key and exercises the whole
pipeline; `anthropic` calls Claude vision for real extraction.
"""
