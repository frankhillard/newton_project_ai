#!/usr/bin/env bash
# batch_track.sh
# --------------------------------------------------------------------------
# Runs ball_tracker.py on every .mp4 file in a directory, writing one
# trajectory CSV per video into an output directory.
#
# Usage:
#   ./batch_track.sh data/videos data/trajectories
#
# For data/videos/P112078_1.mp4 this produces:
#   data/trajectories/P112078_1.csv
#
# Extra flags after the two directories are passed straight through to
# ball_tracker.py, e.g.:
#   ./batch_track.sh data/videos data/trajectories --max-gap-interp 8
# --------------------------------------------------------------------------
set -euo pipefail

VIDEO_DIR="${1:-}"
OUT_DIR="${2:-}"
shift 2 2>/dev/null || true
EXTRA_ARGS=("$@")

if [ -z "$VIDEO_DIR" ] || [ -z "$OUT_DIR" ] || [ ! -d "$VIDEO_DIR" ]; then
    echo "Usage: $0 <video_dir> <output_dir> [extra ball_tracker.py args]" >&2
    echo "  e.g.: $0 data/videos data/trajectories" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

shopt -s nullglob
videos=("$VIDEO_DIR"/*.mp4)

if [ ${#videos[@]} -eq 0 ]; then
    echo "No .mp4 files found in $VIDEO_DIR"
    exit 0
fi

echo "Found ${#videos[@]} video(s) in $VIDEO_DIR"
echo

n_ok=0
n_fail=0
failed_files=()

for video in "${videos[@]}"; do
    base="$(basename "$video" .mp4)"
    out_csv="$OUT_DIR/$base.csv"

    echo "=== $base ==="
    if python3 "$SCRIPT_DIR/ball_tracker.py" --video "$video" --out "$out_csv" "${EXTRA_ARGS[@]}"; then
        n_ok=$((n_ok + 1))
    else
        echo "FAILED: $video" >&2
        n_fail=$((n_fail + 1))
        failed_files+=("$video")
    fi
    echo
done

echo "=================================================="
echo "Done: $n_ok succeeded, $n_fail failed (out of ${#videos[@]})"
if [ $n_fail -gt 0 ]; then
    echo "Failed files:"
    for f in "${failed_files[@]}"; do
        echo "  - $f"
    done
fi
