"""
live.py
-------
Open this computer's camera, track a tennis ball, and draw the discovered
trajectory x(t), y(t) on the live image.

While the ball is in flight the same calibration as calibrate.py (tennis-ball
diameter, y up) and the same discovery as physics_discovery.py run in the
background. The fitted curve is redrawn on the camera image and refreshed
at most every --refit-interval seconds. A miss shorter than about 0.4 s
does not end the throw. After a longer gap the last curve stays on screen
until the next throw has a fit of its own.

Keys
----
    q  quit
    c  clear the current throw and the overlay

Usage
-----
    python live.py
    python live.py --camera 0 --refit-interval 0.2 --max-poly-degree 6
"""
import argparse
import sys
import threading
import time
import warnings

import cv2
import numpy as np

from ball_tracker import DEFAULT_HSV_HIGH, DEFAULT_HSV_LOW, detect_ball_color
from calibrate import TENNIS_BALL_DIAMETER_M
from physics_discovery import discover_axis, predict_library

MIN_POINTS = 8
CONFIRM_HITS = 3
MISS_GAP_SEC = 0.4
SEARCH_RADIUS_PX = 150
EXTRAPOLATE_SEC = 0.4
CURVE_SAMPLES = 200


def evaluate_axis(engine, poly_best, sparse_best, t_grid):
    """Evaluate one discovered axis. Polynomial and sparse winners store
    coefficients on different objects, so the engine name selects which."""
    if engine == "polynomial":
        return np.polyval(poly_best["coeffs"], t_grid)
    return predict_library(t_grid, sparse_best["coeffs_raw"], sparse_best["names"])


def fit_samples(samples, max_poly_degree, throw_id):
    """Calibrate a pixel trajectory and discover x(t), y(t).

    `samples` is an array of rows (t_sec, x_px, y_px, radius_px), with t
    measured from the first point of this throw. The origin of the meter
    frame is that first point, and y increases upward.
    """
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 6:
        raise ValueError("need at least 6 points to discover an equation")

    t = samples[:, 0]
    x_px = samples[:, 1]
    y_px = samples[:, 2]
    radius = samples[:, 3]
    good = radius[np.isfinite(radius) & (radius > 0)]
    if len(good) == 0:
        raise ValueError("no detected radius to calibrate from")

    median_radius = float(np.median(good))
    meters_per_px = TENNIS_BALL_DIAMETER_M / (2.0 * median_radius)
    x0 = float(x_px[0])
    y0 = float(y_px[0])
    x_m = (x_px - x0) * meters_per_px
    y_m = -(y_px - y0) * meters_per_px

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        x_winner, x_poly, x_sparse = discover_axis(
            t, x_m, "x", max_poly_degree=max_poly_degree, verbose=False)
        y_winner, y_poly, y_sparse = discover_axis(
            t, y_m, "y", max_poly_degree=max_poly_degree, verbose=False)

    return {
        "throw_id": throw_id,
        "x0": x0,
        "y0": y0,
        "meters_per_px": meters_per_px,
        "t_max": float(t[-1]),
        "n_points": int(len(samples)),
        "x_engine": x_winner["engine"],
        "x_poly": x_poly,
        "x_sparse": x_sparse,
        "y_engine": y_winner["engine"],
        "y_poly": y_poly,
        "y_sparse": y_sparse,
        "x_equation": x_winner["equation"],
        "y_equation": y_winner["equation"],
    }


def model_to_pixels(model, t_grid):
    """Map discovered x(t), y(t) in meters back onto the camera image."""
    x_m = evaluate_axis(model["x_engine"], model["x_poly"], model["x_sparse"], t_grid)
    y_m = evaluate_axis(model["y_engine"], model["y_poly"], model["y_sparse"], t_grid)
    x_px = model["x0"] + x_m / model["meters_per_px"]
    y_px = model["y0"] - y_m / model["meters_per_px"]
    return x_px, y_px


def curve_polyline(model, width, height, extrapolate_sec, n_samples=CURVE_SAMPLES):
    """Pixel polyline of the fit from t=0 through a short extrapolation.

    Sampling stops at the first point that leaves the frame, so the overlay
    does not wrap around the image border.
    """
    t_end = float(model["t_max"]) + extrapolate_sec
    if t_end <= 0 or width <= 0 or height <= 0:
        return None
    t_grid = np.linspace(0.0, t_end, n_samples)
    x_px, y_px = model_to_pixels(model, t_grid)
    pts = []
    for x, y in zip(x_px, y_px):
        if not (0.0 <= x < width and 0.0 <= y < height):
            break
        pts.append([int(round(float(x))), int(round(float(y)))])
    if len(pts) < 2:
        return None
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)


class FitWorker:
    """Run discovery off the camera thread. Only the latest request is kept."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None
        self._result = None
        self._busy = False
        self._stop = False
        self._wake = threading.Event()
        self._last_error = None
        self._thread = threading.Thread(target=self._run, name="fit-worker", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        with self._lock:
            self._stop = True
            self._pending = None
        self._wake.set()
        self._thread.join(timeout=5.0)

    def submit(self, samples, throw_id, max_poly_degree):
        job = (np.asarray(samples, dtype=float).copy(), int(throw_id), int(max_poly_degree))
        with self._lock:
            self._pending = job
            should_wake = not self._busy
            self._busy = True
        if should_wake:
            self._wake.set()

    def poll(self):
        with self._lock:
            result = self._result
            self._result = None
            return result

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                with self._lock:
                    if self._stop:
                        return
                    job = self._pending
                    self._pending = None
                    if job is None:
                        self._busy = False
                        break
                samples, throw_id, max_poly_degree = job
                model = None
                try:
                    model = fit_samples(samples, max_poly_degree, throw_id)
                except Exception as exc:
                    message = str(exc)
                    if message != self._last_error:
                        print(f"Fit failed: {message}", file=sys.stderr)
                        self._last_error = message
                with self._lock:
                    if model is not None and not self._stop:
                        self._result = model


class ThrowTracker:
    """Accumulate one throw at a time and decide when to ask for a refit."""

    def __init__(self):
        self.throw_id = 0
        self.oldest_acceptable = 0
        self.samples = []
        self.pending = []
        self.tracking = False
        self.t0 = None
        self.last_seen = None
        self.prev_center = None
        self.overlay = None
        self.last_equation_text = None
        self.last_submit_time = -1e9
        self.last_submit_count = -1
        self.last_submit_throw = -1

    def clear(self):
        self.throw_id += 1
        self.oldest_acceptable = self.throw_id
        self.samples = []
        self.pending = []
        self.tracking = False
        self.t0 = None
        self.last_seen = None
        self.prev_center = None
        self.overlay = None
        self.last_equation_text = None
        self.last_submit_count = -1
        self.last_submit_throw = -1

    def update(self, now, detection):
        if detection is None:
            if self.tracking:
                if self.last_seen is not None and (now - self.last_seen) > MISS_GAP_SEC:
                    self.tracking = False
                    self.prev_center = None
                    self.pending.clear()
            else:
                self.pending.clear()
                self.prev_center = None
            return

        x, y, radius = detection
        self.prev_center = (float(x), float(y))
        if self.tracking:
            self.last_seen = now
            self.samples.append((now - self.t0, float(x), float(y), float(radius)))
            return

        self.pending.append((now, float(x), float(y), float(radius)))
        if len(self.pending) >= CONFIRM_HITS:
            self._begin()

    def _begin(self):
        self.throw_id += 1
        self.tracking = True
        self.t0 = self.pending[0][0]
        self.samples = [(t - self.t0, x, y, radius) for t, x, y, radius in self.pending]
        self.last_seen = self.pending[-1][0]
        self.pending.clear()
        self.last_submit_time = -1e9
        self.last_submit_count = -1
        self.last_submit_throw = -1

    def maybe_submit(self, now, worker, max_poly_degree, interval, min_points):
        n = len(self.samples)
        if n < min_points:
            return
        if self.last_submit_throw == self.throw_id and self.last_submit_count == n:
            return
        if (self.tracking and self.last_submit_throw == self.throw_id
                and (now - self.last_submit_time) < interval):
            return
        worker.submit(self.samples, self.throw_id, max_poly_degree)
        self.last_submit_time = now
        self.last_submit_count = n
        self.last_submit_throw = self.throw_id

    def accept(self, model):
        if model is None:
            return
        if model["throw_id"] < self.oldest_acceptable:
            return
        if self.overlay is not None and model["throw_id"] < self.overlay["throw_id"]:
            return
        text = model["x_equation"] + "\n" + model["y_equation"]
        if text != self.last_equation_text:
            print(model["x_equation"])
            print(model["y_equation"])
            self.last_equation_text = text
        self.overlay = model

    def status(self):
        if self.tracking:
            return "tracking"
        if self.samples:
            return "held"
        if self.pending:
            return "arming"
        return "searching"


def _draw_line(frame, text, origin, font_scale=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin
    max_width = max(1, frame.shape[1] - x - 8)
    scale = font_scale
    while scale > 0.35:
        (width, _), _ = cv2.getTextSize(text, font, scale, 1)
        if width <= max_width:
            break
        scale -= 0.05
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return y + 22


def draw_overlay(frame, tracker, detection, extrapolate_sec):
    height, width = frame.shape[:2]
    if tracker.overlay is not None:
        polyline = curve_polyline(tracker.overlay, width, height, extrapolate_sec)
        if polyline is not None:
            cv2.polylines(frame, [polyline], False, (255, 220, 0), 2, cv2.LINE_AA)
    if detection is not None:
        x, y, radius = detection
        cv2.circle(frame, (int(round(x)), int(round(y))), max(1, int(round(radius))), (0, 0, 255), 2)

    y_text = _draw_line(frame, f"{len(tracker.samples)} pts  {tracker.status()}  (q quit, c clear)", (10, 24))
    overlay = tracker.overlay
    if overlay is not None:
        y_text = _draw_line(frame, overlay["x_equation"], (10, y_text))
        _draw_line(frame, overlay["y_equation"], (10, y_text))


def run(camera, refit_interval, max_poly_degree, min_points, extrapolate_sec):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {camera}", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    worker = FitWorker()
    worker.start()
    tracker = ThrowTracker()
    window = "live"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: camera stopped returning frames", file=sys.stderr)
                break

            now = time.monotonic()
            result, _mask = detect_ball_color(
                frame, DEFAULT_HSV_LOW, DEFAULT_HSV_HIGH,
                prev_center=tracker.prev_center, search_radius=SEARCH_RADIUS_PX,
            )
            tracker.update(now, result)
            tracker.maybe_submit(now, worker, max_poly_degree, refit_interval, min_points)
            tracker.accept(worker.poll())
            draw_overlay(frame, tracker, result, extrapolate_sec)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                tracker.clear()
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="Track a tennis ball on the webcam and overlay the discovered x(t), y(t)."
    )
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--refit-interval", type=float, default=0.2,
                     help="Minimum seconds between discovery updates while the ball is in flight")
    ap.add_argument("--max-poly-degree", type=int, default=6,
                     help="Highest polynomial degree considered by discovery (default: 6)")
    ap.add_argument("--min-points", type=int, default=MIN_POINTS,
                     help="Detections required before the first fit of a throw")
    ap.add_argument("--extrapolate", type=float, default=EXTRAPOLATE_SEC,
                     help="Seconds to extend the curve past the latest detection")
    args = ap.parse_args()
    if args.min_points < 6:
        ap.error("--min-points must be at least 6")
    if args.refit_interval < 0 or args.extrapolate < 0:
        ap.error("--refit-interval and --extrapolate must be >= 0")

    run(args.camera, args.refit_interval, args.max_poly_degree, args.min_points, args.extrapolate)


if __name__ == "__main__":
    main()
