"""
ball_tracker.py
----------------
Stage 1 of the Newton pipeline: detect and track a tennis ball across video
frames, producing a CSV of raw pixel-space observations.

Detection strategy
-------------------
Tennis balls are a very distinctive fluorescent yellow-green. We detect
candidate blobs with an HSV color threshold, then pick the most plausible
blob per frame using size + circularity + (once tracking has started)
proximity to the previous known position. This is far more robust for a
fast-moving, motion-blurred ball than Hough circle detection alone, but we
fall back to Hough circles on frames where the color mask fails.

Output
------
A CSV with columns: frame, t_sec, x_px, y_px, radius_px, found
  - frame:     integer frame index (0-based)
  - t_sec:     frame / fps
  - x_px,y_px: ball center in pixel coordinates (origin top-left, y down)
  - radius_px: estimated ball radius in pixels (useful for calibration)
  - found:     1 if the ball was detected this frame, 0 if interpolated/missing

Usage
-----
    python ball_tracker.py --video path/to/video.mp4 --out trajectory.csv

    # Tune the color range interactively if detection is poor:
    python ball_tracker.py --video path/to/video.mp4 --out trajectory.csv --debug
"""
import argparse
import csv
import sys

import cv2
import numpy as np

# Default HSV range for a fluorescent tennis ball (yellow-green).
# Tune these if your lighting/ball differs. Use --debug to see the mask live.
DEFAULT_HSV_LOW = (29, 70, 70)
DEFAULT_HSV_HIGH = (64, 255, 255)

MIN_RADIUS_PX = 3
MAX_RADIUS_PX = 80


def detect_ball_color(frame_bgr, hsv_low, hsv_high, prev_center=None, search_radius=None):
    """Detect the most plausible ball blob via HSV color thresholding.

    Returns (x, y, radius) in pixels, or None if nothing plausible was found.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    candidates = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius < MIN_RADIUS_PX or radius > MAX_RADIUS_PX:
            continue
        area = cv2.contourArea(c)
        circle_area = np.pi * radius * radius
        if circle_area <= 0:
            continue
        circularity = area / circle_area  # 1.0 = perfect circle
        if circularity < 0.4:  # reject very non-circular blobs (motion streaks etc.)
            continue
        score = circularity
        if prev_center is not None:
            dist = np.hypot(x - prev_center[0], y - prev_center[1])
            if search_radius is not None and dist > search_radius:
                continue
            score -= dist / 1000.0  # mild penalty for jumping far
        candidates.append((score, x, y, radius))

    if not candidates:
        return None, mask

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, radius = candidates[0]
    return (x, y, radius), mask


def detect_ball_hough(frame_bgr):
    """Fallback detector using Hough circle transform on the grayscale frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=30, minRadius=MIN_RADIUS_PX, maxRadius=MAX_RADIUS_PX,
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype(int)
    x, y, r = circles[0]
    return (float(x), float(y), float(r))


def track_video(video_path, out_csv, hsv_low=DEFAULT_HSV_LOW, hsv_high=DEFAULT_HSV_HIGH,
                 debug=False, max_gap_interp=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}  fps={fps:.3f}  frames~={n_frames_hint}")

    rows = []  # each: [frame, t_sec, x, y, r, found]
    prev_center = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, mask = detect_ball_color(frame, hsv_low, hsv_high, prev_center,
                                          search_radius=150)
        if result is None:
            hough = detect_ball_hough(frame)
            if hough is not None:
                x, y, r = hough
                found = 1
            else:
                x, y, r = np.nan, np.nan, np.nan
                found = 0
        else:
            x, y, r = result
            found = 1

        if found:
            prev_center = (x, y)

        rows.append([frame_idx, frame_idx / fps, x, y, r, found])

        if debug:
            disp = frame.copy()
            if found:
                cv2.circle(disp, (int(x), int(y)), int(r), (0, 0, 255), 2)
            cv2.imshow("frame", disp)
            cv2.imshow("mask", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if debug:
        cv2.destroyAllWindows()

    n_found = sum(r[5] for r in rows)
    print(f"Detected ball in {n_found}/{len(rows)} frames "
          f"({100.0 * n_found / max(1, len(rows)):.1f}%)")

    _interpolate_gaps(rows, max_gap=max_gap_interp)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_sec", "x_px", "y_px", "radius_px", "found"])
        writer.writerows(rows)
    print(f"Wrote trajectory to {out_csv}")


def _interpolate_gaps(rows, max_gap=5):
    """Linearly interpolate short runs of missing detections in place.

    `found` stays 0 for interpolated points so downstream stages can decide
    whether to trust them, but x/y/r become usable numeric estimates instead
    of NaN, which keeps the trajectory continuous for plotting/fitting.
    """
    n = len(rows)
    i = 0
    while i < n:
        if rows[i][5] == 1:
            i += 1
            continue
        j = i
        while j < n and rows[j][5] == 0:
            j += 1
        gap_len = j - i
        if i > 0 and j < n and gap_len <= max_gap:
            x0, y0, r0 = rows[i - 1][2], rows[i - 1][3], rows[i - 1][4]
            x1, y1, r1 = rows[j][2], rows[j][3], rows[j][4]
            for k in range(i, j):
                t = (k - (i - 1)) / (j - (i - 1))
                rows[k][2] = x0 + t * (x1 - x0)
                rows[k][3] = y0 + t * (y1 - y0)
                rows[k][4] = r0 + t * (r1 - r0)
        i = j


def main():
    ap = argparse.ArgumentParser(description="Track a tennis ball in a video and export its pixel trajectory.")
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--out", default="trajectory.csv", help="Path to output CSV")
    ap.add_argument("--hsv-low", nargs=3, type=int, default=DEFAULT_HSV_LOW, metavar=("H", "S", "V"))
    ap.add_argument("--hsv-high", nargs=3, type=int, default=DEFAULT_HSV_HIGH, metavar=("H", "S", "V"))
    ap.add_argument("--debug", action="store_true", help="Show live detection windows")
    ap.add_argument("--max-gap-interp", type=int, default=5,
                     help="Max consecutive missing frames to linearly interpolate")
    args = ap.parse_args()

    track_video(args.video, args.out, tuple(args.hsv_low), tuple(args.hsv_high),
                debug=args.debug, max_gap_interp=args.max_gap_interp)


if __name__ == "__main__":
    main()
