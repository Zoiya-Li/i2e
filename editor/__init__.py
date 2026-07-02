"""Node ③ — the editor. A thin browser surface for rendering the IR and editing
the high-value fields (text / localization / bbox). It owns NO moat logic: on
save it hands the edited IR to the Python backend, which runs the canonical
`capture_diff` (Node ⑤). One implementation of the moat, never duplicated in JS.
"""
