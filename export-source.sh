#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/i2e-source-export.zip}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Exporting source-only archive to: $OUT"

cd "$ROOT"
git ls-files 2>/dev/null | while IFS= read -r f; do
    case "$f" in
        .git/*) continue ;;
        work/diagram2ppt/archive_*/*|work/diagram2ppt/v3_out_*/*|work/diagram2ppt/v2_out*/*|work/diagram2ppt/v22_out*/*|work/diagram2ppt/v24_out*/*) continue ;;
        work/hf_cache/*|work/ms_cache/*) continue ;;
        work/poster/*|work/poster_sd15_0739/*) continue ;;
        work/gen_decompose/output/*|work/derender_IMG_9493/*|work/ocr_venv/*|work/test_run/*) continue ;;
        *.log|*.assets/*|__pycache__/*|*.pyc|.pytest_cache/*|.DS_Store) continue ;;
        mobile_sam.pt|sam3.pt) continue ;;
        framework.png.ocr_upscale*.png|test.png.ocr_upscale.png) continue ;;
        IMG_9493.jpg) continue ;;
    esac
    printf '%s\n' "$f"
done > "$TMPDIR/includes.txt"

mkdir -p "$TMPDIR/stage/i2e"
rsync -a --files-from="$TMPDIR/includes.txt" . "$TMPDIR/stage/i2e/"

# Remove any empty work/test_run directory that git ls-files might have staged
rm -rf "$TMPDIR/stage/i2e/work/test_run"

cd "$TMPDIR/stage"
zip -rq "$OUT" i2e

echo "Done. Source-only archive: $OUT"
