"""
newton_fit.py
-------------
Stage 3 of the Newton pipeline: given a calibrated (x_m, y_m) vs t_sec
trajectory, DISCOVER the equation of motion from the data itself, with no
assumption of projectile motion, constant acceleration, or any other
specific physical law baked into the fitting code.

Why this is different from "just curve_fit a parabola"
--------------------------------------------------------
A naive approach fits x(t) and y(t) against a hand-picked formula such as
  x(t) = x0 + vx0*t
  y(t) = y0 + vy0*t - 0.5*g*t^2
which already presupposes constant horizontal velocity and constant
vertical acceleration (i.e. it presupposes Newton's second law under
uniform gravity with no drag). That's circular if the goal is to discover
the law, not confirm it.

Instead, this module treats x(t) and y(t) independently as unknown
functions and asks two complementary, assumption-free questions:

1. "What polynomial degree best explains this coordinate?"
   Fit polynomials of increasing degree (0, 1, 2, 3, ...) and pick the
   degree that minimizes the Bayesian Information Criterion (BIC), which
   penalizes extra parameters so the method doesn't just pick the highest
   degree (which would always fit noise better). If the ball is really
   under constant acceleration, degree 2 should win for y(t) on its own,
   without that being assumed anywhere in the code.

2. "Which terms, from a broad dictionary of candidate functions, are
   actually needed?"
   Build a library of candidate basis functions of t (constant, powers of
   t, square root of t, several exponential-decay rates to allow for
   drag-like terms), then run sparse regression (sequential thresholded
   least squares, the core trick behind the SINDy method for discovering
   governing equations from data) to zero out every term that isn't
   needed. What survives IS the discovered equation, printed in
   human-readable form.

Both methods are run independently for x(t) and y(t) (they need not
produce the same kind of equation - e.g. x(t) might turn out linear while
y(t) turns out quadratic, or vice versa, or something else entirely; the
code does not know in advance). The method with the better
(lower) BIC is reported as the best discovered model per axis.

Any resemblance of a discovered y(t) quadratic coefficient to -0.5 * 9.81
is noted only as a post-hoc observation for interpretation, never as
something the fitting procedure assumed or searched for specifically.

Output
------
- Printed discovered equations for x(t) and y(t), from both engines, with
  R^2 and BIC, so you can see which terms survived and how good the fit is.
- A PNG figure: trajectory, x(t) fit, y(t) fit, residuals - for the best
  discovered model per axis.
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G_REFERENCE = 9.81  # m/s^2, Earth's gravity - used ONLY for post-hoc comparison/annotation


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


def bic(y_actual, y_predicted, n_params):
    n = len(y_actual)
    rss = np.sum((y_actual - y_predicted) ** 2)
    rss = max(rss, 1e-300)  # avoid log(0)
    return n * np.log(rss / n) + n_params * np.log(n)


# ---- Engine 1: polynomial order selection via BIC --------------------------

def fit_best_polynomial(t, y, max_degree=6):
    """Fit polynomials of degree 0..max_degree independently; return the one
    that minimizes BIC, plus all candidates (for transparency/plotting)."""
    n = len(t)
    max_degree = min(max_degree, n - 2)  # need enough points to fit + judge
    candidates = []
    for deg in range(0, max_degree + 1):
        coeffs = np.polyfit(t, y, deg)  # highest power first
        y_pred = np.polyval(coeffs, t)
        candidates.append({
            "degree": deg,
            "coeffs": coeffs,
            "y_pred": y_pred,
            "r2": r_squared(y, y_pred),
            "bic": bic(y, y_pred, deg + 1),
        })
    best = min(candidates, key=lambda c: c["bic"])
    return best, candidates


def polynomial_equation_str(coeffs, var="x"):
    """coeffs are highest-degree-first (numpy.polyfit convention)."""
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-9:
            continue
        if power == 0:
            terms.append(f"{c:+.5f}")
        elif power == 1:
            terms.append(f"{c:+.5f}*t")
        else:
            terms.append(f"{c:+.5f}*t^{power}")
    if not terms:
        terms = ["0"]
    return f"{var}(t) = " + " ".join(terms).lstrip("+")


# ---- Engine 2: sparse symbolic regression over a function library ----------
# (sequential thresholded least squares - the core mechanism behind SINDy,
#  Brunton/Proctor/Kutz 2016, for discovering governing equations from data)

def build_library(t, extra_exp_rates=(0.5, 1.0, 2.0, 5.0, 10.0)):
    """Build a dictionary of candidate basis functions of t. This is
    intentionally broad and physics-agnostic: constant, several polynomial
    powers, square root, and a spread of exponential-decay rates (to allow
    discovery of drag-like terms without assuming a specific drag
    coefficient). Sparse regression below decides which of these, if any,
    actually belong in the equation.
    """
    names = ["1", "t", "t^2", "t^3", "t^4"]
    cols = [np.ones_like(t), t, t ** 2, t ** 3, t ** 4]

    t_safe = np.clip(t, 0, None)
    names.append("sqrt(t)")
    cols.append(np.sqrt(t_safe))

    for k in extra_exp_rates:
        names.append(f"exp(-{k:g}t)")
        cols.append(np.exp(-k * t))

    Theta = np.column_stack(cols)
    return Theta, names


def _ridge_lstsq(A, b, ridge_lambda):
    """Ridge-regularized least squares: solves (A^T A + lambda*I) x = A^T b.

    A library of polynomial + exponential-decay basis functions is often
    severely ill-conditioned (e.g. over a short time window, t^3, t^4 and
    several exp(-k*t) curves are nearly linear combinations of each other).
    Plain least-squares on a near-singular matrix can return huge,
    canceling coefficients that technically minimize training error but are
    numerically meaningless - not a discovered equation, just noise
    amplification. A small ridge penalty keeps the solve stable without
    materially affecting genuinely well-identified coefficients.
    """
    ATA = A.T @ A
    reg = ridge_lambda * np.eye(ATA.shape[0])
    ATb = A.T @ b
    return np.linalg.solve(ATA + reg, ATb)


def sequential_thresholded_least_squares(Theta, y, threshold, n_iters=15, ridge_lambda=1e-4):
    """Core SINDy sparsification step: start from ridge-regularized least
    squares, repeatedly zero out small coefficients and refit on the
    survivors, until the sparsity pattern stabilizes."""
    Xi = _ridge_lstsq(Theta, y, ridge_lambda)
    for _ in range(n_iters):
        small = np.abs(Xi) < threshold
        if np.all(small):
            Xi[:] = 0
            break
        Xi[small] = 0
        big = ~small
        Xi[big] = _ridge_lstsq(Theta[:, big], y, ridge_lambda)
    return Xi


def discover_sparse_equation(t, y, max_terms_cap=None):
    """Sweep sparsity thresholds, pick the sparsity pattern that minimizes
    BIC (not just training error), so the method is penalized for spurious
    extra terms the same way the polynomial-order engine is."""
    Theta_raw, names = build_library(t)
    n = len(t)

    # Standardize columns for numerically stable thresholding; rescale
    # coefficients back to raw units afterwards.
    norms = np.linalg.norm(Theta_raw, axis=0)
    norms[norms == 0] = 1.0
    Theta = Theta_raw / norms

    thresholds = np.logspace(-3, 1, 60)
    best = None
    for thresh in thresholds:
        Xi = sequential_thresholded_least_squares(Theta.copy(), y, thresh)
        k = int(np.count_nonzero(Xi))
        if k == 0:
            continue
        if max_terms_cap is not None and k > max_terms_cap:
            continue
        y_pred = Theta @ Xi
        score = bic(y, y_pred, k)
        if best is None or score < best["bic"]:
            best = {"Xi": Xi.copy(), "k": k, "bic": score, "y_pred": y_pred,
                     "threshold": thresh}

    if best is None:
        # fallback: dense least squares, no sparsification possible
        Xi, *_ = np.linalg.lstsq(Theta, y, rcond=None)
        y_pred = Theta @ Xi
        best = {"Xi": Xi, "k": len(Xi), "bic": bic(y, y_pred, len(Xi)),
                 "y_pred": y_pred, "threshold": 0.0}

    coeffs_raw = best["Xi"] / norms
    best["coeffs_raw"] = coeffs_raw
    best["names"] = names
    best["r2"] = r_squared(y, best["y_pred"])
    return best


def sparse_equation_str(coeffs_raw, names, var="x"):
    terms = []
    for c, name in zip(coeffs_raw, names):
        if abs(c) < 1e-9:
            continue
        if name == "1":
            terms.append(f"{c:+.5f}")
        else:
            terms.append(f"{c:+.5f}*{name}")
    if not terms:
        terms = ["0"]
    return f"{var}(t) = " + " ".join(terms).lstrip("+")


def predict_library(t, coeffs_raw, names):
    Theta_raw, built_names = build_library(t)
    assert built_names == names
    return Theta_raw @ coeffs_raw


# ---- Combine both engines per axis -----------------------------------------

def discover_axis(t, y, var_name, max_poly_degree=6):
    poly_best, poly_candidates = fit_best_polynomial(t, y, max_degree=max_poly_degree)
    sparse_best = discover_sparse_equation(t, y)

    poly_eq = polynomial_equation_str(poly_best["coeffs"], var=var_name)
    sparse_eq = sparse_equation_str(sparse_best["coeffs_raw"], sparse_best["names"], var=var_name)

    print(f"\n[{var_name}(t)] Engine 1 - polynomial order selection (BIC):")
    print(f"  best degree = {poly_best['degree']}   R^2 = {poly_best['r2']:.5f}   BIC = {poly_best['bic']:.2f}")
    print(f"  {poly_eq}")
    ranked = sorted(poly_candidates, key=lambda c: c["bic"])[:3]
    if len(ranked) > 1:
        print("  (closest competing degrees, for transparency - small BIC gaps mean the "
              "data doesn't strongly distinguish between them):")
        for c in ranked:
            marker = "<- chosen" if c["degree"] == poly_best["degree"] else ""
            print(f"    degree {c['degree']}: BIC = {c['bic']:.2f}   R^2 = {c['r2']:.5f}  {marker}")

    print(f"[{var_name}(t)] Engine 2 - sparse symbolic regression (STLSQ dictionary):")
    print(f"  nonzero terms = {sparse_best['k']}   R^2 = {sparse_best['r2']:.5f}   BIC = {sparse_best['bic']:.2f}")
    print(f"  {sparse_eq}")

    if poly_best["bic"] <= sparse_best["bic"]:
        winner = {"engine": "polynomial", "y_pred": poly_best["y_pred"],
                   "r2": poly_best["r2"], "bic": poly_best["bic"], "equation": poly_eq}
    else:
        winner = {"engine": "sparse", "y_pred": sparse_best["y_pred"],
                   "r2": sparse_best["r2"], "bic": sparse_best["bic"], "equation": sparse_eq}
    print(f"[{var_name}(t)] => best discovered model: {winner['engine']}   {winner['equation']}")

    # Post-hoc physical interpretation, computed from whatever was
    # discovered - never assumed in advance. Only triggered if a t^2 term
    # is present in the polynomial engine's winning model.
    if poly_best["degree"] >= 2:
        c_t2 = poly_best["coeffs"][-3]  # coefficient of t^2, highest-power-first indexing
        implied_accel = 2 * c_t2
        note = (f"  Note: the discovered t^2 coefficient implies a constant second "
                f"derivative (acceleration) of {implied_accel:.3f} m/s^2 in {var_name}(t).")
        if abs(abs(implied_accel) - G_REFERENCE) < 1.0:
            note += f" This is close to Earth's gravitational acceleration ({G_REFERENCE} m/s^2)."
        print(note)

    return winner, poly_best, sparse_best


# ---- Reporting & plotting ---------------------------------------------------

def plot_results(t, x, y, x_winner, y_winner, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.scatter(x, y, s=10, color="black", label="observed", zorder=3)
    ax.plot(x_winner["y_pred"], y_winner["y_pred"], color="crimson",
            label=f"discovered ({x_winner['engine']}/{y_winner['engine']})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory")
    ax.legend(fontsize=8)
    ax.axis("equal")

    ax = axes[0, 1]
    ax.scatter(t, x, s=10, color="black", label="observed", zorder=3)
    ax.plot(t, x_winner["y_pred"], color="crimson", label=f"discovered ({x_winner['engine']})")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x (m)")
    ax.set_title("Horizontal position vs time")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(t, y, s=10, color="black", label="observed", zorder=3)
    ax.plot(t, y_winner["y_pred"], color="crimson", label=f"discovered ({y_winner['engine']})")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("y (m)")
    ax.set_title("Vertical position vs time")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, y - y_winner["y_pred"], marker="o", markersize=3, color="crimson", label="y residual")
    ax.plot(t, x - x_winner["y_pred"], marker="o", markersize=3, color="steelblue", label="x residual")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("residual (m)")
    ax.set_title("Fit residuals (discovered models)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nWrote plot to {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description="Discover the equation of motion from a calibrated tennis-ball "
                     "trajectory, with no assumed physical model."
    )
    ap.add_argument("--in", dest="in_csv", required=True, help="Calibrated trajectory CSV (from calibrate.py)")
    ap.add_argument("--out-plot", default="trajectory_fit.png", help="Output plot PNG")
    ap.add_argument("--only-found", action="store_true",
                     help="Fit only using confidently-detected frames (exclude interpolated ones)")
    ap.add_argument("--max-poly-degree", type=int, default=6,
                     help="Highest polynomial degree to consider in Engine 1 (default: 6)")
    args = ap.parse_args()

    t, x, y, found = load_calibrated_csv(args.in_csv, only_found=args.only_found)
    if len(t) < 6:
        raise SystemExit("Not enough data points to discover an equation (need >= 6).")

    print("Discovering x(t) ...")
    x_winner, _, _ = discover_axis(t, x, "x", max_poly_degree=args.max_poly_degree)
    print("\nDiscovering y(t) ...")
    y_winner, _, _ = discover_axis(t, y, "y", max_poly_degree=args.max_poly_degree)

    plot_results(t, x, y, x_winner, y_winner, args.out_plot)


if __name__ == "__main__":
    main()
