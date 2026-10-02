"""
newton_fit.py
-------------
Stage 3 of the Newton pipeline: given a calibrated (x_m, y_m) vs t_sec
trajectory, fit candidate physical models and report which best explains
the data - i.e. "discover" the law of motion from observation, the way
Newton would have wanted.

Models fit
----------
1. Free-fall / projectile motion (no air resistance):
     x(t) = x0 + vx0 * t                          (constant horizontal velocity)
     y(t) = y0 + vy0 * t - 0.5 * g * t^2           (constant vertical acceleration)
   Fitting y(t) as a quadratic directly recovers an estimate of g, the
   quantity most people mean by "discovering the law of motion" here.

2. Projectile motion with linear air drag (optional, for comparison):
     A quadratic-drag or linear-drag ODE is solved numerically and fit to
     the data via least squares over (vx0, vy0, drag coefficient). This
     model is included so you can see whether drag meaningfully improves
     the fit for your footage (fast, light tennis balls do show some drag).

Outputs
-------
- Printed fit parameters: g (m/s^2), initial velocity components, R^2 for
  each model, so you can see how close the recovered g is to the
  true value (9.81 m/s^2) and whether drag is detectable.
- A PNG figure with: (a) trajectory scatter + fitted curves, (b) x(t) fit,
  (c) y(t) fit, (d) residuals, so you can visually judge fit quality.
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint

G_TRUE = 9.81  # m/s^2, for reference/comparison only


def load_calibrated_csv(path, only_found=False):
    """Load a calibrated trajectory CSV, always dropping rows with NaN
    x/y (long detection gaps that calibrate.py could not interpolate, since
    ball_tracker.py only interpolates gaps up to --max-gap-interp frames).
    Fitting functions cannot handle NaNs, so these rows are removed
    regardless of the `only_found` setting.
    """
    t, x, y, found = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t_sec"]))
            x.append(float(row["x_m"]))
            y.append(float(row["y_m"]))
            found.append(int(row["found"]))
    t, x, y, found = map(np.array, (t, x, y, found))

    n_total = len(t)
    nan_mask = np.isnan(x) | np.isnan(y)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        print(f"Dropping {n_nan}/{n_total} rows with NaN x/y "
              f"(detection gaps longer than ball_tracker.py's --max-gap-interp). "
              f"If this is a large fraction of the data, re-run ball_tracker.py "
              f"with a larger --max-gap-interp, or improve detection (--debug).")
        t, x, y, found = t[~nan_mask], x[~nan_mask], y[~nan_mask], found[~nan_mask]

    if only_found:
        mask = found == 1
        n_dropped = len(t) - int(mask.sum())
        if n_dropped > 0:
            print(f"--only-found: dropping {n_dropped} additional interpolated rows.")
        t, x, y = t[mask], x[mask], y[mask]

    return t, x, y, found


def r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---- Model 1: no-drag projectile motion -----------------------------------

def linear_model(t, x0, v0):
    return x0 + v0 * t


def quadratic_model(t, y0, vy0, g):
    return y0 + vy0 * t - 0.5 * g * t ** 2


def fit_no_drag(t, x, y):
    popt_x, _ = curve_fit(linear_model, t, x, p0=[x[0], 0.0])
    x0, vx0 = popt_x

    popt_y, _ = curve_fit(quadratic_model, t, y, p0=[y[0], 0.0, G_TRUE])
    y0, vy0, g = popt_y

    x_pred = linear_model(t, *popt_x)
    y_pred = quadratic_model(t, *popt_y)

    return {
        "name": "No-drag projectile motion",
        "params": {"x0": x0, "vx0": vx0, "y0": y0, "vy0": vy0, "g": g},
        "x_pred": x_pred, "y_pred": y_pred,
        "r2_x": r_squared(x, x_pred), "r2_y": r_squared(y, y_pred),
    }


# ---- Model 2: projectile motion with linear air drag -----------------------

def _drag_ode(state, t, k):
    """state = [x, y, vx, vy]; k = drag coefficient / mass (1/s)."""
    x, y, vx, vy = state
    speed = np.hypot(vx, vy)
    ax = -k * speed * vx
    ay = -G_TRUE - k * speed * vy
    return [vx, vy, ax, ay]


def _simulate_drag(t, x0, y0, vx0, vy0, k):
    state0 = [x0, y0, vx0, vy0]
    sol = odeint(_drag_ode, state0, t, args=(k,))
    return sol[:, 0], sol[:, 1]


def fit_with_drag(t, x, y):
    def residuals(params):
        x0, y0, vx0, vy0, k = params
        x_pred, y_pred = _simulate_drag(t, x0, y0, vx0, vy0, max(k, 0.0))
        return np.concatenate([x_pred - x, y_pred - y])

    # initial guess from finite differences
    vx0_guess = (x[1] - x[0]) / max(t[1] - t[0], 1e-6)
    vy0_guess = (y[1] - y[0]) / max(t[1] - t[0], 1e-6)
    p0 = [x[0], y[0], vx0_guess, vy0_guess, 0.01]

    from scipy.optimize import least_squares
    result = least_squares(residuals, p0, method="lm", max_nfev=5000)
    x0, y0, vx0, vy0, k = result.x
    x_pred, y_pred = _simulate_drag(t, x0, y0, vx0, vy0, max(k, 0.0))

    return {
        "name": "Projectile motion with quadratic air drag",
        "params": {"x0": x0, "y0": y0, "vx0": vx0, "vy0": vy0, "drag_k": k},
        "x_pred": x_pred, "y_pred": y_pred,
        "r2_x": r_squared(x, x_pred), "r2_y": r_squared(y, y_pred),
    }


# ---- Reporting & plotting ---------------------------------------------------

def print_report(model):
    print(f"\n--- {model['name']} ---")
    for k, v in model["params"].items():
        print(f"  {k:8s} = {v:.5f}")
    print(f"  R^2 (x) = {model['r2_x']:.5f}   R^2 (y) = {model['r2_y']:.5f}")
    if "g" in model["params"]:
        g = model["params"]["g"]
        err_pct = 100.0 * abs(g - G_TRUE) / G_TRUE
        print(f"  => recovered g = {g:.3f} m/s^2  (true g = {G_TRUE}, error = {err_pct:.1f}%)")


def plot_results(t, x, y, models, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) trajectory in space
    ax = axes[0, 0]
    ax.scatter(x, y, s=10, color="black", label="observed", zorder=3)
    for m in models:
        ax.plot(m["x_pred"], m["y_pred"], label=m["name"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory")
    ax.legend(fontsize=8)
    ax.axis("equal")

    # (b) x(t)
    ax = axes[0, 1]
    ax.scatter(t, x, s=10, color="black", label="observed", zorder=3)
    for m in models:
        ax.plot(t, m["x_pred"], label=m["name"])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x (m)")
    ax.set_title("Horizontal position vs time")
    ax.legend(fontsize=8)

    # (c) y(t)
    ax = axes[1, 0]
    ax.scatter(t, y, s=10, color="black", label="observed", zorder=3)
    for m in models:
        ax.plot(t, m["y_pred"], label=m["name"])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("y (m)")
    ax.set_title("Vertical position vs time")
    ax.legend(fontsize=8)

    # (d) residuals (y) for the best model
    ax = axes[1, 1]
    for m in models:
        ax.plot(t, y - m["y_pred"], marker="o", markersize=3, label=m["name"])
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("residual y (m)")
    ax.set_title("Vertical fit residuals")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nWrote plot to {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Fit physics laws to a calibrated tennis-ball trajectory.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Calibrated trajectory CSV (from calibrate.py)")
    ap.add_argument("--out-plot", default="trajectory_fit.png", help="Output plot PNG")
    ap.add_argument("--only-found", action="store_true",
                     help="Fit only using confidently-detected frames (exclude interpolated ones)")
    ap.add_argument("--skip-drag-model", action="store_true",
                     help="Skip the (slower) drag-model fit")
    args = ap.parse_args()

    t, x, y, found = load_calibrated_csv(args.in_csv, only_found=args.only_found)
    if len(t) < 5:
        raise SystemExit("Not enough data points to fit a model (need >= 5).")

    models = [fit_no_drag(t, x, y)]
    if not args.skip_drag_model:
        try:
            models.append(fit_with_drag(t, x, y))
        except Exception as e:
            print(f"Drag-model fit failed ({e}); continuing with no-drag model only.")

    for m in models:
        print_report(m)

    plot_results(t, x, y, models, args.out_plot)


if __name__ == "__main__":
    main()
