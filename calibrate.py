"""
calibrate.py
------------
Stage 2 of the Newton pipeline: convert pixel measurements to real-world
metric units (meters), and flip the y-axis so "up" is positive (as physics
convention expects), since image coordinates have y increasing downward.

Two calibration strategies are supported:

1. Ball-diameter calibration (default, no manual setup needed):
   A regulation tennis ball is 6.54-6.86 cm in diameter (ITF spec); we use
   6.7 cm as the midpoint. We take the median detected radius_px across all
   frames where the ball was actually found (not interpolated) and use that
   as the pixel-to-meter scale. This works well as long as the ball is
   roughly perpendicular to the camera's viewing axis (limited depth
   motion) and the camera has negligible lens distortion for the ball's
   region of the frame.

2. Manual reference calibration:
   If you can measure a known real-world length in the frame (e.g. distance
   between two court line markings visible in the shot), supply
   --ref-px and --ref-m to compute the scale directly. This is more
   accurate for scenes with significant depth (toward/away from camera).

Output
------
A CSV with columns: frame, t_sec, x_m, y_m, found
  - x_m, y_m: position in meters, in a coordinate system with x increasing
              rightward and y increasing UPWARD, origin at the ball's first
              detected position.
"""
import argparse
import csv

import numpy as np

TENNIS_BALL_DIAMETER_M = 0.067  # ITF regulation: 6.54-6.86 cm, midpoint used


def load_trajectory_csv(path):
    frames, t, x, y, r, found = [], [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            t.append(float(row["t_sec"]))
            x.append(float(row["x_px"]))
            y.append(float(row["y_px"]))
            r.append(float(row["radius_px"]))
            found.append(int(row["found"]))
    return (np.array(frames), np.array(t), np.array(x), np.array(y),
            np.array(r), np.array(found))


def scale_from_ball_radius(radius_px, found, ball_diameter_m=TENNIS_BALL_DIAMETER_M):
    """meters-per-pixel, using the median radius across confidently-detected frames."""
    good = radius_px[found == 1]
    good = good[~np.isnan(good)]
    if len(good) == 0:
        raise ValueError("No confidently-detected frames to calibrate from. "
                          "Use manual --ref-px/--ref-m calibration instead.")
    median_radius_px = float(np.median(good))
    meters_per_px = ball_diameter_m / (2.0 * median_radius_px)
    print(f"Calibration: median ball radius = {median_radius_px:.2f}px "
          f"-> {meters_per_px * 1000:.4f} mm/px")
    return meters_per_px


def scale_from_reference(ref_px, ref_m):
    meters_per_px = ref_m / ref_px
    print(f"Calibration: reference {ref_px}px = {ref_m}m "
          f"-> {meters_per_px * 1000:.4f} mm/px")
    return meters_per_px


def calibrate(trajectory_csv, out_csv, meters_per_px=None,
              ref_px=None, ref_m=None, ball_diameter_m=TENNIS_BALL_DIAMETER_M):
    frames, t, x_px, y_px, r_px, found = load_trajectory_csv(trajectory_csv)

    if meters_per_px is None:
        if ref_px is not None and ref_m is not None:
            meters_per_px = scale_from_reference(ref_px, ref_m)
        else:
            meters_per_px = scale_from_ball_radius(r_px, found, ball_diameter_m)

    # Origin at first point; flip y so "up" is positive (physics convention).
    x0, y0 = x_px[0], y_px[0]
    x_m = (x_px - x0) * meters_per_px
    y_m = -(y_px - y0) * meters_per_px

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_sec", "x_m", "y_m", "found"])
        for i in range(len(frames)):
            writer.writerow([frames[i], t[i], x_m[i], y_m[i], found[i]])
    print(f"Wrote calibrated trajectory to {out_csv}")
    return meters_per_px


def main():
    ap = argparse.ArgumentParser(description="Convert a pixel trajectory CSV to real-world meters.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input trajectory CSV (from ball_tracker.py)")
    ap.add_argument("--out", default="trajectory_m.csv", help="Output calibrated CSV")
    ap.add_argument("--ref-px", type=float, default=None, help="Manual calibration: reference length in pixels")
    ap.add_argument("--ref-m", type=float, default=None, help="Manual calibration: same reference length in meters")
    ap.add_argument("--ball-diameter-m", type=float, default=TENNIS_BALL_DIAMETER_M,
                     help="Ball diameter in meters (default: regulation tennis ball)")
    args = ap.parse_args()

    calibrate(args.in_csv, args.out, ref_px=args.ref_px, ref_m=args.ref_m,
              ball_diameter_m=args.ball_diameter_m)


if __name__ == "__main__":
    main()
