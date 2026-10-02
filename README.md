# Newton Project AI — Discovering Laws of Motion from Tennis Ball Video

Track a tennis ball across video frames, calibrate the measurements to
real-world units, and fit physical models to the trajectory to recover
quantities like gravitational acceleration `g` directly from observation —
the same empirical approach Newton himself used, applied to a camera instead
of a telescope.

## Pipeline

```
video.mp4
   │  ball_tracker.py   (HSV color detection + Hough fallback)
   ▼
trajectory.csv          (frame, t_sec, x_px, y_px, radius_px, found)
   │  calibrate.py       (pixels → meters, via known ball diameter)
   ▼
trajectory_m.csv        (frame, t_sec, x_m, y_m, found)
   │  physics_discovery.py (discover the equation of motion - no assumed model)
   ▼
fitted parameters (g, v0, ...) + trajectory_fit.png
```

Each stage is a standalone script with its own CSV in/out, so you can
inspect, correct, or swap out any stage independently (e.g. plug in a
different, better ball detector without touching the physics fitting code).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Track the ball

```bash
python ball_tracker.py --video my_throw.mp4 --out trajectory.csv
```

If detection looks poor (wrong ball color range for your lighting/footage),
run with `--debug` to see the live color mask and tune `--hsv-low` /
`--hsv-high`:

```bash
python ball_tracker.py --video my_throw.mp4 --out trajectory.csv --debug
```

### 2. Calibrate to real-world units

By default, calibration uses the known diameter of a regulation tennis ball
(6.7 cm) and the median detected radius in pixels:

```bash
python calibrate.py --in trajectory.csv --out trajectory_m.csv
```

If your shot has significant depth (ball moving toward/away from the
camera), ball-diameter calibration will be less accurate. Instead, measure a
known real-world length visible in frame (e.g. court line spacing) and use:

```bash
python calibrate.py --in trajectory.csv --out trajectory_m.csv \
    --ref-px 342 --ref-m 1.37
```

### 3. Discover the law of motion

```bash
python physics_discovery.py --in trajectory_m.csv --out-plot trajectory_fit.png
```

Unlike a script that fits a pre-written formula (e.g. assuming
`y(t) = y0 + vy0*t - 0.5*g*t^2`), `physics_discovery.py` makes **no assumption**
about the motion model. It discovers the equation from the data using two
independent, assumption-free methods, run separately for `x(t)` and `y(t)`
(they don't have to turn out to be the same kind of equation):

1. **Polynomial order selection (BIC)** — fits polynomials of increasing
   degree and picks the degree that minimizes the Bayesian Information
   Criterion, which penalizes unnecessary extra terms. If the true motion
   is constant-acceleration, degree 2 should win for `y(t)` on its own —
   the code never tells it to look for a parabola specifically.
2. **Sparse symbolic regression (STLSQ dictionary)** — builds a broad
   library of candidate terms (powers of `t`, `sqrt(t)`, several
   exponential-decay rates to allow drag-like behavior) and uses sequential
   thresholded least squares (the core mechanism behind the SINDy method
   for discovering governing equations from data) to zero out every term
   that isn't needed. What survives is printed as a human-readable equation.

The method with the better (lower) BIC is reported as the best discovered
model per axis, and if a `t²` term is discovered, the implied constant
acceleration is reported as a post-hoc observation (compared against
Earth's gravity only for interpretation, never assumed in the fit itself).

The script also prints the next-closest competing polynomial degrees and
their BIC scores, so you can see when the data doesn't strongly
distinguish between, say, a clean quadratic and a slightly-better-fitting
quartic — that's a sign of noisy or gappy tracking data, not necessarily a
more complex true law.

### 4. Newtonian fitting

```bash
python newton_fit.py --in trajectory_m.csv --out-plot trajectory_fit.png
```

A script that fits a pre-written formula (e.g. assuming
`y(t) = y0 + vy0*t - 0.5*g*t^2`), `physics_discovery.py` makes **assumption**
about the motion model.

It recovers the gravitationnal constant g

## Validating the pipeline

`physics_discovery.py` was checked against synthetic data it had no way to
"know" the generating equation for in advance:

- **No-drag projectile motion** (`x` linear, `y` quadratic, true `g = 9.81`):
  it independently discovered degree 1 for `x(t)` and degree 2 for `y(t)`,
  recovering `g` to within 0.1–0.4% depending on noise/gaps.
- **Pure constant-velocity motion** (no acceleration in either axis): it
  correctly discovered degree 1 for _both_ axes and did **not** fabricate a
  spurious quadratic/gravity term — confirming the method adapts to the
  data rather than defaulting to a parabola.

Real-world accuracy will additionally depend on tracking quality and
calibration; the "closest competing degrees" output helps you judge that
on your own footage.

## Notes & limitations

- **Detection**: HSV color thresholding works well for a standard
  fluorescent tennis ball against a reasonably uncluttered background.
  Fast motion blur, similarly-colored backgrounds, or occlusion will hurt
  detection rate — check the `found` column in `trajectory.csv`.
- **Interpolation**: short gaps (≤5 frames by default) in detection are
  linearly interpolated so the trajectory stays continuous; longer gaps are
  left as `found=0` with the last-known values.
- **Calibration accuracy**: ball-diameter calibration assumes the ball
  stays roughly perpendicular to the camera's line of sight. For throws
  with a lot of depth motion, use manual reference calibration instead.
- **Frame rate**: motion fitting is only as good as your camera's FPS —
  higher frame rate footage gives a cleaner recovered `g`.

## Repository structure

```
ball_tracker.py         # Stage 1: video -> pixel trajectory CSV
calibrate.py             # Stage 2: pixel trajectory -> metric trajectory CSV
newton_fit.py           # Optional Stage: metric trajectory -> recover g (bias newton)
physics_discovery.py    # Stage 3: metric trajectory -> discovered equation of motion + plot
requirements.txt
README.md
```
