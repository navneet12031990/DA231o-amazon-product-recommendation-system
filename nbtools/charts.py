"""Charts for the notebooks, drawn only from files the pipeline writes (no number is typed in here).

Every function takes the local results folder (a copy of data.results_prefix), saves a PNG into
<results>/charts/ and returns the matplotlib figure, or returns None and prints why when its input
file does not exist yet.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, DARK = "#2f6db5", "#e07b1f", "#9aa3ad", "#1d2433"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def _load(results: str, name: str):
    path = os.path.join(results, name)
    if not os.path.exists(path):
        print(f"[charts] {name} not found in {results}: skipped (run the step that writes it first)")
        return None
    return json.load(open(path)) if name.endswith(".json") else pd.read_csv(path)


def _save(fig, results: str, name: str):
    os.makedirs(os.path.join(results, "charts"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(results, "charts", name), dpi=150, bbox_inches="tight")
    return fig


def _fmt(n) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


# ------------------------------------------------------------------ Module 1: data engineering
def data_funnel(results: str):
    prof = _load(results, "data_profile.json")
    if prof is None:
        return None
    f = prof["funnel"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [s["stage"] for s in f][::-1]
    vals = [s["rows"] for s in f][::-1]
    bars = ax.barh(labels, vals, color=[GREY, GREY, BLUE, BLUE, BLUE, DARK][::-1][:len(vals)])
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:,}", va="center", fontsize=9)
    ax.set(xlabel="rows", title=f"ETL funnel ({len(prof['categories'])} categories)")
    ax.grid(axis="y", alpha=0)
    ax.set_xlim(0, max(vals) * 1.25)
    return _save(fig, results, "data_funnel.png")


def data_categories(results: str):
    prof = _load(results, "data_profile.json")
    if prof is None:
        return None
    pc = pd.DataFrame(prof["per_category"]).T.sort_values("raw_rows")
    fig, axes = plt.subplots(1, 2, figsize=(11, max(2.8, 0.38 * len(pc) + 1.5)))
    y = np.arange(len(pc))
    axes[0].barh(y - 0.2, pc["raw_rows"], 0.4, color=GREY, label="raw rows")
    axes[0].barh(y + 0.2, pc["final_rows"], 0.4, color=BLUE, label="after ETL")
    axes[0].set_yticks(y, pc.index); axes[0].set(xlabel="ratings", title="Ratings per category")
    axes[0].legend(loc="lower right"); axes[0].grid(axis="y", alpha=0)
    axes[1].barh(y - 0.2, pc["users"], 0.4, color=ORANGE, label="users")
    axes[1].barh(y + 0.2, pc["items"], 0.4, color=DARK, label="items")
    axes[1].set_yticks(y, [""] * len(pc)); axes[1].set(xlabel="count", title="Users and items after ETL")
    axes[1].legend(loc="lower right"); axes[1].grid(axis="y", alpha=0)
    return _save(fig, results, "data_categories.png")


def data_distributions(results: str):
    prof = _load(results, "data_profile.json")
    if prof is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    r = prof["ratings"]
    stars = sorted(r, key=float)
    axes[0].bar([f"{float(s):g}★" for s in stars], [r[s] for s in stars], color=BLUE)
    axes[0].set(title="Rating distribution", ylabel="ratings")
    for ax, key, title, color in ((axes[1], "user_activity", "Users by training ratings", ORANGE),
                                  (axes[2], "item_activity", "Items by training ratings", DARK)):
        b = prof[key]
        ax.bar([x["bucket"] for x in b], [x["count"] for x in b], color=color)
        ax.set_yscale("log"); ax.set(title=title, xlabel="ratings in train", ylabel="count (log)")
        ax.tick_params(axis="x", rotation=45)
    cw, t = prof["cold_warm_users"], prof["totals"]
    axes[1].text(0.98, 0.95, f"cold (<{cw['threshold']}): {cw['cold']:,}\nwarm: {cw['warm']:,}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
    axes[2].text(0.98, 0.95, f"density {t['density']:.2e}", transform=axes[2].transAxes, ha="right", va="top", fontsize=9)
    return _save(fig, results, "data_distributions.png")


# ------------------------------------------------------------------ Module 2: accuracy
def accuracy(results: str):
    c = _load(results, "centralized_metrics.json")
    cw = _load(results, "coldstart_metrics.json")
    if c is None or cw is None:
        return None
    groups = ["overall", "cold users", "warm users"]
    model = [c["rmse"], cw["cold"]["rmse"], cw["warm"]["rmse"]]
    bias = [c["baselines"]["bias"]["rmse"], cw["cold"]["baselines"]["bias"]["rmse"], cw["warm"]["baselines"]["bias"]["rmse"]]
    glob_ = [c["baselines"]["global_mean"]["rmse"], cw["cold"]["baselines"]["global_mean"]["rmse"],
             cw["warm"]["baselines"]["global_mean"]["rmse"]]
    x = np.arange(3)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    for k, (vals, lab, col) in enumerate(((glob_, "global mean", GREY), (bias, "bias baseline", ORANGE),
                                          (model, "bias + ALS (model)", BLUE))):
        bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, groups); ax.set(ylabel="RMSE (lower is better)", title="Centralized accuracy on the test set")
    ax.set_ylim(min(model + bias) * 0.85, max(glob_) * 1.06); ax.legend(fontsize=8)
    p = [cw["cold"]["precision_at_k"], cw["warm"]["precision_at_k"]]
    ax2.bar(["cold users", "warm users"], p, color=[ORANGE, BLUE])
    for i, v in enumerate(p):
        ax2.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    ax2.set(title="precision@10 (sampled users)", ylabel="precision@10")
    return _save(fig, results, "accuracy.png")


def accuracy_by_category(results: str):
    c = _load(results, "centralized_metrics.json")
    if c is None or len(c.get("by_category", {})) < 2:
        if c is not None:
            print("[charts] one category only: per-category chart skipped")
        return None
    bc = pd.DataFrame(c["by_category"]).T.sort_values("n_test_rows")
    y = np.arange(len(bc))
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.4 * len(bc) + 1.2)))
    ax.barh(y - 0.2, bc["bias_rmse"], 0.4, color=ORANGE, label="bias baseline")
    ax.barh(y + 0.2, bc["rmse"], 0.4, color=BLUE, label="bias + ALS")
    ax.set_yticks(y, [f"{i} ({_fmt(n)} test)" for i, n in zip(bc.index, bc["n_test_rows"])])
    ax.set(xlabel="RMSE", title="Accuracy per category", xlim=(min(bc["rmse"].min(), bc["bias_rmse"].min()) * 0.9, None))
    ax.legend(); ax.grid(axis="y", alpha=0)
    return _save(fig, results, "accuracy_by_category.png")


# ------------------------------------------------------------------ Module 3: federated
def federated(results: str):
    conv = _load(results, "federated_convergence.csv")
    comp = _load(results, "comparison.json")
    if conv is None or comp is None:
        return None
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.8))
    for n, g in conv.groupby("num_clients"):
        g = g.dropna(subset=["rmse"])
        a1.plot(g["round"], g["rmse"], "o-", ms=3, label=f"{n} client{'s' if n > 1 else ''}")
    a1.axhline(comp["centralized"]["rmse"], color=DARK, ls="--", lw=1, label="centralized")
    a1.set(xlabel="communication round", ylabel="test RMSE", title="FedAvg convergence (round 0 = bias only)")
    a1.legend(fontsize=8)
    f = comp["federated"]
    ns = sorted(f, key=int)
    x = np.arange(len(ns))
    cold = [f[n]["final_rmse_cold"] for n in ns]; warm = [f[n]["final_rmse_warm"] for n in ns]
    a2.bar(x - 0.2, cold, 0.4, color=ORANGE, label="cold")
    a2.bar(x + 0.2, warm, 0.4, color=BLUE, label="warm")
    a2.axhline(comp["centralized"]["rmse_cold"], color=ORANGE, ls="--", lw=1)
    a2.axhline(comp["centralized"]["rmse_warm"], color=BLUE, ls="--", lw=1)
    vals = cold + warm + [comp["centralized"]["rmse_cold"], comp["centralized"]["rmse_warm"]]
    a2.set_xticks(x, ns)
    a2.set(xlabel="clients", ylabel="final RMSE", title="Cold / warm (dashed = centralized)",
           ylim=(min(vals) * 0.97, max(vals) * 1.02))
    a2.legend(fontsize=8)
    mb = [f[n]["total_bytes"] / 1e6 for n in ns]
    a3.bar(x, mb, color=DARK)
    for i, v in enumerate(mb):
        a3.text(i, v, f"{v:.1f} MB" if v >= 1 else f"{v * 1e3:.0f} kB", ha="center", va="bottom", fontsize=8)
    a3.set_xticks(x, ns)
    a3.set(xlabel="clients", ylabel="MB (10^6 bytes)", title="Simulated communication, all rounds")
    a1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle("Simulated FedAvg — no privacy guarantee is provided or implied", fontsize=9, y=1.02)
    return _save(fig, results, "federated.png")


# ------------------------------------------------------------------ Module 4: scaling
def _scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    med = (df.groupby(["mode", "sweep", "executors"], as_index=False)
             .agg(time_sec=("time_sec", "median"), runs=("time_sec", "size"), rows=("rows", "median"),
                  granted=("executors_granted", "min")))
    out = []
    for (mode, sweep), g in med.groupby(["mode", "sweep"]):
        g = g.sort_values("executors")
        n0, t0 = g["executors"].iloc[0], g["time_sec"].iloc[0]
        for _, r in g.iterrows():
            n, t = r["executors"], r["time_sec"]
            if sweep == "strong":
                speed = t0 / t
                eff = speed * n0 / n
            else:
                speed = (n / n0) * t0 / t
                eff = t0 / t
            out.append({"mode": mode, "sweep": sweep, "executors": int(n), "granted": int(r["granted"]),
                        "rows": int(r["rows"]), "runs": int(r["runs"]), "median_time_sec": round(t, 1),
                        "speedup": round(speed, 2), "ideal_speedup": round(n / n0, 2), "efficiency": round(eff, 2)})
    return pd.DataFrame(out)


def scaling(results: str, threshold: float = 0.5):
    """Strong + weak scaling for every mode in scaling_results.csv, with a verdict per curve.
    'Achieved' here means parallel efficiency >= `threshold` at the largest executor count
    (a stated rule of thumb, printed with the numbers so the reader can judge)."""
    df = _load(results, "scaling_results.csv")
    if df is None or df.empty:
        return None
    tab = _scaling_table(df)
    modes = sorted(tab["mode"].unique())
    fig, axes = plt.subplots(len(modes), 3, figsize=(15, 3.6 * len(modes)), squeeze=False)
    verdicts = []
    for i, mode in enumerate(modes):
        s = tab[(tab["mode"] == mode) & (tab["sweep"] == "strong")]
        w = tab[(tab["mode"] == mode) & (tab["sweep"] == "weak")]
        ax_t, ax_s, ax_w = axes[i]
        for d, lab, col in ((s, "strong (fixed data)", BLUE), (w, "weak (data grows with executors)", ORANGE)):
            if len(d):
                ax_t.plot(d["executors"], d["median_time_sec"], "o-", color=col, label=lab)
        ax_t.set(xlabel="executors", ylabel="median time (s)", title=f"{mode}: training time", ylim=(0, None))
        ax_t.legend(fontsize=8)
        if len(s) >= 2:
            n = s["executors"].to_numpy()
            ax_s.plot(n, s["speedup"], "o-", color=BLUE, label="measured speedup")
            ax_s.plot(n, s["ideal_speedup"], "--", color=GREY, label="ideal")
            ax2 = ax_s.twinx(); ax2.plot(n, s["efficiency"], "s:", color=DARK, label="efficiency"); ax2.set_ylim(0, 1.15)
            ax2.set_ylabel("efficiency"); ax2.grid(False)
            last = s.iloc[-1]
            ok = last["efficiency"] >= threshold
            verdicts.append(f"{mode} STRONG scaling {'ACHIEVED' if ok else 'limited'}: {last['speedup']}x speedup "
                            f"at {last['executors']} executors (ideal {last['ideal_speedup']}x), efficiency {last['efficiency']}")
            ax_s.set(xlabel="executors", ylabel="speedup", title=f"{mode}: strong scaling — {'achieved' if ok else 'limited'}")
            ax_s.legend(fontsize=8, loc="upper left")
        else:
            ax_s.text(0.5, 0.5, "strong sweep not run", ha="center", transform=ax_s.transAxes); ax_s.set_axis_off()
        if len(w) >= 2:
            n = w["executors"].to_numpy()
            ax_w.plot(n, w["efficiency"], "o-", color=ORANGE, label="weak-scaling efficiency T(n0)/T(n)")
            ax_w.axhline(1, ls="--", color=GREY, label="ideal (constant time)")
            last = w.iloc[-1]
            ok = last["efficiency"] >= threshold
            verdicts.append(f"{mode} WEAK scaling {'ACHIEVED' if ok else 'limited'}: time x{1 / last['efficiency']:.2f} "
                            f"while data grew x{last['ideal_speedup']} ({w['rows'].iloc[0]:,} -> {last['rows']:,} rows), "
                            f"efficiency {last['efficiency']}")
            ax_w.set(xlabel="executors", ylabel="efficiency", ylim=(0, 1.15),
                     title=f"{mode}: weak scaling — {'achieved' if ok else 'limited'}")
            ax_w.legend(fontsize=8)
        else:
            ax_w.text(0.5, 0.5, "weak sweep not run", ha="center", transform=ax_w.transAxes); ax_w.set_axis_off()
    print(f"Verdict rule: efficiency >= {threshold} at the largest executor count. Medians over repetitions.")
    for v in verdicts:
        print("  " + v)
    short = tab[tab["granted"] < tab["executors"]]
    if len(short):
        print("WARNING: some runs got fewer executors than requested:\n", short)
    tab.to_csv(os.path.join(results, "scaling_summary.csv"), index=False)
    return _save(fig, results, "scaling_overview.png"), tab


def size_scaling(runs_log_csv: str, results: str):
    """Test notebook: job wall time per stage against dataset size, accumulated over runs."""
    if not os.path.exists(runs_log_csv):
        print("[charts] no runs log yet"); return None
    df = pd.read_csv(runs_log_csv)
    df = df[df["stage"].isin(["ETL (convert + build)", "ETL build", "centralized", "federated", "profile"])]
    if df["run_tag"].nunique() < 2:
        print("[charts] size-scaling chart needs at least two runs with different category sets "
              f"(runs so far: {df['run_tag'].nunique()})")
    fig, ax = plt.subplots(figsize=(8, 4))
    for stage, g in df.groupby("stage"):
        g = g.sort_values("raw_rows")
        ax.plot(g["raw_rows"], g["seconds"], "o-", label=stage)
        for _, r in g.iterrows():
            ax.annotate(f"{int(r['n_categories'])} cat, {int(r['workers'])}w", (r["raw_rows"], r["seconds"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="raw ratings in the run", ylabel="job wall time (s, incl. job start-up)",
           title="Pipeline time vs dataset size (test runs)")
    ax.legend(fontsize=8)
    return _save(fig, results, "size_scaling.png")


# ------------------------------------------------------------------ Module 4: streaming
def load_stream(results: str):
    files = glob.glob(os.path.join(results, "streaming", "batches", "*.json"))
    batches = pd.DataFrame([json.load(open(f)) for f in files]).sort_values("batch_id") if files else pd.DataFrame()
    lat_files = glob.glob(os.path.join(results, "streaming_latency", "*.csv"))
    lat = pd.concat([pd.read_csv(f) for f in lat_files]) if lat_files else pd.DataFrame()
    summary = json.load(open(os.path.join(results, "streaming_summary.json"))) \
        if os.path.exists(os.path.join(results, "streaming_summary.json")) else None
    return batches, lat, summary


def streaming(results: str):
    batches, lat, summary = load_stream(results)
    if batches.empty:
        print("[charts] no streaming batches found: skipped"); return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    a = axes[0, 0]
    a.bar(batches["batch_id"], batches["scored"], color=BLUE, label="scored")
    a.bar(batches["batch_id"], batches["dropped_unknown"], bottom=batches["scored"], color=ORANGE, label="unknown id (dropped)")
    a.set(xlabel="micro-batch", ylabel="events", title="Events per micro-batch"); a.legend(fontsize=8)
    a = axes[0, 1]
    a.plot(batches["batch_id"], batches["processing_sec"], "o-", ms=3, color=DARK)
    a.set(xlabel="micro-batch", ylabel="seconds", title="Batch processing time")
    a = axes[1, 0]
    if not lat.empty:
        a.hist(lat["latency_sec"], bins=40, color=BLUE)
        for key, q, ls in (("p50", 0.5, "--"), ("p95", 0.95, ":")):
            v = summary["latency_sec"][key] if summary else lat["latency_sec"].quantile(q)   # same numbers as the summary
            a.axvline(v, color=DARK, ls=ls, label=f"{key} = {v:.2f}s")
        a.set(xlabel="latency (scored time - Kafka record time), s", ylabel="events", title="End-to-end latency")
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "latency CSV not written yet (run the summary step)", ha="center", transform=a.transAxes)
    a = axes[1, 1]
    if not lat.empty:
        t = np.sort(lat["scored_ts"].to_numpy()); t0 = lat["kafka_ts"].min()
        a.plot(t - t0, np.arange(1, len(t) + 1), color=BLUE, label="scored")
        p = np.sort(lat["kafka_ts"].to_numpy())
        a.plot(p - t0, np.arange(1, len(p) + 1), color=GREY, ls="--", label="published to Kafka")
        a.set(xlabel="seconds since first event", ylabel="cumulative events", title="Publish vs scoring progress")
        a.legend(fontsize=8)
    if summary:
        fig.suptitle(f"Kafka -> Spark Structured Streaming, model: {summary['model']} | "
                     f"{summary['events_scored']:,} scored, latency p50 {summary['latency_sec']['p50']:.2f}s / "
                     f"p95 {summary['latency_sec']['p95']:.2f}s, throughput {summary['scoring_throughput_per_sec']:.1f}/s",
                     fontsize=10)
    return _save(fig, results, "streaming.png")


def all_charts(results: str, show=None) -> list:
    """Draw every chart whose input exists; returns the figures (Colab shows them inline)."""
    figs = []
    for fn in (data_funnel, data_categories, data_distributions, accuracy, accuracy_by_category, federated, streaming):
        f = fn(results)
        if f is not None:
            figs.append(f)
    s = scaling(results)
    if s is not None:
        figs.append(s[0])
    return figs
