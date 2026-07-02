"""Isolated CLI for true PowerPoint render/compare.

This module intentionally stays small: it imports the PowerPoint snapshot
stack, not the PPTX builder or planner, so it can be launched as a clean
subprocess after a deck has been built.
"""
from __future__ import annotations

import argparse
import sys

from work.diagram2ppt.v2.snapshot import compare, snapshot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="isolated PowerPoint render helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render PPTX to PNG")
    r.add_argument("pptx")
    r.add_argument("out")
    r.add_argument("--dpi", type=int, default=130)

    c = sub.add_parser("compare", help="stack original and rendered PPTX")
    c.add_argument("pptx")
    c.add_argument("original")
    c.add_argument("out")
    c.add_argument("--dpi", type=int, default=130)

    args = ap.parse_args(argv)
    if args.cmd == "render":
        print(snapshot(args.pptx, args.out, dpi=args.dpi))
        return 0
    if args.cmd == "compare":
        print(compare(args.pptx, args.original, args.out, dpi=args.dpi))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
