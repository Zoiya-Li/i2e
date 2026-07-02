"""North-star metric: average human corrections per image.

The single number that proves the flywheel is turning: on a FIXED benchmark set,
this must DROP over time as the extraction model improves. If it doesn't drop,
we built a tool, not a flywheel.

    python bench/flywheel.py <dir-of-*.ir.json>
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def measure(ir_dir: str) -> dict:
    # matches both predicted (<name>.ir.json) and shipped (<name>.edited.ir.json) files
    paths = sorted(Path(ir_dir).glob("*.ir.json"))
    if not paths:
        return {"images": 0, "corrections_per_image": 0.0, "by_kind": {}}
    # one entry per image id; prefer the version carrying the most corrections (the edited save)
    best: dict[str, dict] = {}
    for p in paths:
        doc = json.loads(p.read_text())
        iid = doc.get("id", str(p))
        if iid not in best or len(doc.get("corrections", [])) > len(best[iid].get("corrections", [])):
            best[iid] = doc
    docs = list(best.values())
    total = 0
    by_kind: collections.Counter = collections.Counter()
    for doc in docs:
        corrs = doc.get("corrections", [])
        total += len(corrs)
        by_kind.update(c["kind"] for c in corrs)
    n = len(docs)
    return {
        "images": n,
        "corrections_per_image": round(total / n, 3),
        "by_kind": dict(by_kind.most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python bench/flywheel.py <dir-of-*.ir.json>", file=sys.stderr)
        return 2
    m = measure(argv[0])
    print(f"images:                 {m['images']}")
    print(f"corrections / image:    {m['corrections_per_image']}   <-- north star (must drop over time)")
    print(f"corrections by kind:    {m['by_kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
