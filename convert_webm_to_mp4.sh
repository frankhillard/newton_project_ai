#!/usr/bin/env bash
# convert_webm_to_mp4.sh
# --------------------------------------------------------------------------
# Batch-converts all .webm files in a directory to H.264/MP4, which OpenCV's
# bundled FFmpeg decodes far more reliably than VP8/VP9 WebM.
#
# Usage:
#   ./convert_webm_to_mp4.sh data/videos
#
# Converted files are written alongside the originals as <name>.mp4.
# Originals are left untouched. Already-converted files are skipped unless
# you pass --force.
# --------------------------------------------------------------------------
set -euo pipefail

if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Install it first, e.g.:" >&2
    echo "  macOS:   brew install ffmpeg" >&2
    echo "  Ubuntu:  sudo apt install ffmpeg" >&2
    exit 1
fi

DIR="${1:-}"
FORCE=false
for arg in "$@"; do
    if [ "$arg" == "--force" ]; then
        FORCE=true
    fi
done

if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    echo "Usage: $0 <directory> [--force]" >&2
    echo "  e.g.: $0 data/videos" >&2
    exit 1
fi

shopt -s nullglob
webm_files=("$DIR"/*.webm)

if [ ${#webm_files[@]} -eq 0 ]; then
    echo "No .webm files found in $DIR"
    exit 0
fi

echo "Found ${#webm_files[@]} .webm file(s) in $DIR"

for src in "${webm_files[@]}"; do
    out="${src%.webm}.mp4"
    if [ -f "$out" ] && [ "$FORCE" != true ]; then
        echo "SKIP (already exists): $out"
        continue
    fi
    echo "Converting: $src -> $out"
    ffmpeg -y -loglevel error -i "$src" \
        -c:v libx264 -preset medium -crf 18 \
        -c:a aac -b:a 128k \
        "$out"
done

echo "Done. Converted files are in $DIR/*.mp4"
