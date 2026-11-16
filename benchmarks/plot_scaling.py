"""Module 4 — scaling plots (SPEC.md Section 5). Runs on the client (Colab / Cloud Shell) on the
downloaded results/scaling_results.csv; needs only pandas, numpy and matplotlib.

Strong scaling (fixed data): speedup S(n) = T(n0) / T(n) and efficiency S(n) * n0 / n, relative to
the smallest executor count n0, against ideal and a fitted Amdahl curve
    S(n) = (f + (1 - f) / n0) / (f + (1 - f) / n)         f = serial fraction.
Weak scaling (data grows with n): time T(n) against the ideal constant T(n0), and weak-scaling
efficiency E(n) = T(n0) / T(n) against the ideal 1.0. No Gustafson curve is fitted: here the work
per executor is not constant (distinct items grow more slowly than ratings), which pins a
one-parameter fit at its 0 bound (SPEC.md Section 5).
Every point is the median over repetitions ("measured" when each point is a single run).
"""
import os
import sys

import numpy as np
import pandas as pd

FIT_GRID = np.linspace(0.0, 1.0, 10001)  # serial-fraction values tried by the least-squares fit


def _amdahl(n, n0, f):
    return (f + (1 - f) / n0) / (f + (1 - f) / n)


def _fit(fn, n, n0, measured):
    errs = [np.sum((fn(n, n0, f) - measured) ** 2) for f in FIT_GRID]
    return float(FIT_GRID[int(np.argmin(errs))])


def _measured_label(runs) -> str:
    k = int(max(runs))
    return "measured" if k <= 1 else f"measured (median of {k})"


def medians(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["mode", "sweep", "executors"], as_index=False)
              .agg(time_sec=("time_sec", "median"), runs=("time_sec", "size"),
                   rows=("rows", "median"), executors_granted=("executors_granted", "min")))


def plot_scaling_results(csv_path: str, out_dir: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    short = df[df["executors_granted"] < df["executors"]]
    if len(short):
        print(f"[plot] WARNING: {len(short)} runs got fewer executors than requested:\n{short}")
    med = medians(df)
    os.makedirs(out_dir, exist_ok=True)
    paths, summary = [], []

    for mode in sorted(med["mode"].unique()):
        s = med[(med["mode"] == mode) & (med["sweep"] == "strong")].sort_values("executors")
        if len(s) >= 2:
            n = s["executors"].to_numpy(float)
            t = s["time_sec"].to_numpy(float)
            n0, t0 = n[0], t[0]
            speed = t0 / t
            f = _fit(_amdahl, n, n0, speed)
            nn = np.linspace(n0, n[-1], 200)
            fig, ax = plt.subplots(figsize=(6, 4))
            lab = _measured_label(s["runs"])
            ax.plot(n, speed, "o-", label=lab)
            ax.plot(nn, nn / n0, "--", label="ideal (linear)")
            ax.plot(nn, _amdahl(nn, n0, f), ":", label=f"Amdahl fit, serial fraction f={f:.3f}")
            ax.set(xlabel="executors", ylabel=f"speedup vs {int(n0)} executors",
                   title=f"Strong scaling: {mode} ({int(s['rows'].iloc[0]):,} rows)")
            ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
            p = os.path.join(out_dir, f"speedup_{mode}.png"); fig.savefig(p, dpi=150); plt.close(fig)
            paths.append(p)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(n, speed * n0 / n, "o-", label=lab)
            ax.plot(nn, np.ones_like(nn), "--", label="ideal")
            ax.plot(nn, _amdahl(nn, n0, f) * n0 / nn, ":", label=f"Amdahl fit, f={f:.3f}")
            ax.set(xlabel="executors", ylabel="parallel efficiency", ylim=(0, 1.1),
                   title=f"Strong-scaling efficiency: {mode}")
            ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
            p = os.path.join(out_dir, f"efficiency_{mode}.png"); fig.savefig(p, dpi=150); plt.close(fig)
            paths.append(p)
            summary.append(f"{mode} strong: speedup at {int(n[-1])} executors {speed[-1]:.2f}x "
                           f"(ideal {n[-1] / n0:.0f}x), Amdahl serial fraction {f:.3f}")

        w = med[(med["mode"] == mode) & (med["sweep"] == "weak")].sort_values("executors")
        if len(w) >= 2:
            n = w["executors"].to_numpy(float)
            t = w["time_sec"].to_numpy(float)
            n0, t0 = n[0], t[0]
            eff = t0 / t
            lab = _measured_label(w["runs"])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(n, t, "o-", label=lab)
            ax1.axhline(t0, ls="--", color="gray", label="ideal (constant time)")
            ax1.set(xlabel="executors", ylabel="time (s)", ylim=(0, None),
                    title=f"Weak scaling: {mode} time")
            ax1.legend(); ax1.grid(alpha=.3)
            ax2.plot(n, eff, "o-", label=lab)
            ax2.axhline(1.0, ls="--", color="gray", label="ideal")
            ax2.set(xlabel="executors", ylabel=f"efficiency T({int(n0)}) / T(n)",
                    ylim=(0, max(1.2, float(eff.max()) * 1.1)),
                    title=f"Weak-scaling efficiency: {mode}")
            ax2.legend(); ax2.grid(alpha=.3); fig.tight_layout()
            p = os.path.join(out_dir, f"weak_scaling_{mode}.png"); fig.savefig(p, dpi=150); plt.close(fig)
            paths.append(p)
            summary.append(f"{mode} weak: time {t0:.0f}s -> {t[-1]:.0f}s from {int(n0)} to {int(n[-1])} "
                           f"executors; efficiency " + ", ".join(f"{int(x)}: {e:.2f}" for x, e in zip(n, eff)))

    for line in summary:
        print(f"[plot] {line}")
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python benchmarks/plot_scaling.py <scaling_results.csv> <out_dir>")
    for p in plot_scaling_results(sys.argv[1], sys.argv[2]):
        print(p)
