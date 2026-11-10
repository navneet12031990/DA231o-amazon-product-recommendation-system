"""Streaming dashboard for Navneet's Module 4 outputs (streaming/stream_scorer.py), Colab side.

live(...)       while the scorer job runs: every few seconds, sync results/streaming/batches/*.json
                (written once per micro-batch) and redraw events, throughput and batch times.
build_html(...) after summarize: one self-contained HTML page (KPI tiles + embedded charts) from
                streaming_summary.json, the per-event latency CSV and the batch files.
Read-only: it never touches the stream, the checkpoint or the model.
"""
import base64
import datetime
import html
import io
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np

from nbtools import charts


def _sync(results_prefix: str, local: str) -> None:
    dst = os.path.join(local, "streaming", "batches")
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["gcloud", "storage", "rsync", "-r", results_prefix.rstrip("/") + "/streaming/batches", dst],
                   capture_output=True)


def _live_figure(b, elapsed: float, state: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    t = b["scored_ts"] - b["scored_ts"].min()
    axes[0].plot(t, b["scored"].cumsum(), "o-", ms=3, color=charts.BLUE)
    axes[0].set(xlabel="seconds since first batch", ylabel="events scored", title="Cumulative events scored")
    rate = b["scored"] / b["processing_sec"].clip(lower=1e-9)
    axes[1].bar(b["batch_id"], b["scored"], color=charts.BLUE, label="scored")
    axes[1].bar(b["batch_id"], b["dropped_unknown"], bottom=b["scored"], color=charts.ORANGE, label="unknown dropped")
    axes[1].set(xlabel="micro-batch", ylabel="events", title="Events per micro-batch"); axes[1].legend(fontsize=8)
    axes[2].plot(b["batch_id"], b["processing_sec"], "o-", ms=3, color=charts.DARK)
    axes[2].set(xlabel="micro-batch", ylabel="seconds", title="Batch processing time")
    fig.suptitle(f"LIVE  scorer job: {state} | {len(b)} batches | {int(b['scored'].sum()):,} scored, "
                 f"{int(b['dropped_unknown'].sum()):,} unknown | median batch {b['processing_sec'].median():.1f}s | "
                 f"batch processing rate ~{rate.median():.0f} events/s | {elapsed:.0f}s elapsed", fontsize=10)
    fig.tight_layout()
    return fig


def live(cluster, scorer_job: str, results_prefix: str, local: str, every_sec: int = 10, max_minutes: float = 90,
         watch_job: str = None):
    """Redraw until the scorer job leaves RUNNING (it stops itself after idle_stop_seconds).
    If `watch_job` (the producer) fails, the scorer would wait for events until max_run_seconds, so
    the scorer is cancelled and the live view stops with a message."""
    from IPython.display import clear_output, display
    t0 = time.time()
    state = cluster.job_state(scorer_job)
    while True:
        state = cluster.job_state(scorer_job)
        if watch_job and cluster.job_state(watch_job) in ("ERROR", "CANCELLED"):
            subprocess.run(["gcloud", "dataproc", "jobs", "kill", scorer_job, f"--region={cluster.region}", "--quiet"],
                           capture_output=True)
            print(f"producer job {watch_job} failed: scorer cancelled. See Cloud console > Dataproc > Jobs > {watch_job}")
            return "PRODUCER_FAILED"
        _sync(results_prefix, local)
        b, _, _ = charts.load_stream(local)
        clear_output(wait=True)
        if b.empty:
            print(f"scorer job {scorer_job}: {state}; waiting for the first micro-batch ({time.time() - t0:.0f}s)")
        else:
            fig = _live_figure(b, time.time() - t0, state)
            display(fig); plt.close(fig)
        if state not in ("PENDING", "SETUP_DONE", "RUNNING", ""):
            print(f"scorer job finished: {state}")
            return state
        if time.time() - t0 > max_minutes * 60:
            print("live view stopped (time limit); the scorer keeps running"); return state
        time.sleep(every_sec)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(local: str, out_html: str, title: str = "Kafka streaming dashboard") -> str:
    b, lat, s = charts.load_stream(local)
    if s is None or b.empty:
        raise FileNotFoundError("streaming_summary.json or batch files missing: run the summarize step first")
    tiles = [("Events scored", f"{s['events_scored']:,}", f"of {s['events_received']:,} received"),
             ("Unknown ids dropped", f"{s['dropped_unknown']:,}", "user or item not in the model"),
             ("Duplicate event ids", f"{s['duplicate_event_ids']}", "0 = idempotent output"),
             ("Latency p50", f"{s['latency_sec']['p50']:.2f} s", f"p95 {s['latency_sec']['p95']:.2f} s, "
                                                                f"p99 {s['latency_sec']['p99']:.2f} s"),
             ("Throughput", f"{s['scoring_throughput_per_sec']:.1f} /s", s["throughput_window"]),
             ("Publish rate", f"{(s['publish_rate_per_sec'] or 0):.1f} /s", f"replay target {s['replay_rate_per_sec']}/s"),
             ("Micro-batches", f"{s['batches']}", f"processing p50 {s['batch_processing_sec']['p50']:.1f} s, "
                                                  f"max {s['batch_processing_sec']['max']:.1f} s"),
             ("Streamed RMSE", f"{s['rmse_streamed']:.4f}", f"model: {s['model']}")]
    imgs = []
    fig = charts.streaming(local)
    if fig is not None:
        imgs.append(("Events, batch times, latency and progress", _png(fig)))
    if not lat.empty:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.sort(lat["latency_sec"].to_numpy())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=charts.BLUE)
        ax.set(xlabel="latency (s)", ylabel="share of events", title="Latency CDF")
        fig.tight_layout()
        imgs.append(("Latency CDF", _png(fig)))
    tile_html = "".join(f'<div class="t"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
                        f'<div class="s">{html.escape(sub)}</div></div>' for k, v, sub in tiles)
    img_html = "".join(f'<figure><img alt="{html.escape(c)}" src="data:image/png;base64,{d}"><figcaption>'
                       f'{html.escape(c)}</figcaption></figure>' for c, d in imgs)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1d2433;--mut:#5a6473;--line:#e3e6ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12161d;--card:#1b212b;--ink:#e8ecf1;--mut:#a3adba;--line:#2a323e}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 4px}}
p.m{{color:var(--mut);margin:0 0 18px}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
.t{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}
.k{{color:var(--mut);font-size:13px}}.v{{font-size:26px;font-weight:650;margin:4px 0;font-variant-numeric:tabular-nums}}
.s{{color:var(--mut);font-size:12px}}figure{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px;margin:16px 0}}
img{{max-width:100%;height:auto;display:block}}figcaption{{color:#5a6473;font-size:13px;margin-top:6px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="m">Kafka (broker on the Dataproc master) → Spark Structured Streaming → saved {html.escape(s['model'])} model.
Latency = scored time − Kafka record timestamp. Trigger every {s['trigger_seconds']} s.
Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} from streaming_summary.json, the latency CSV and {len(b)} batch files.
FedAvg models are simulated; no privacy guarantee is provided or implied.</p>
<div class="g">{tile_html}</div>{img_html}</main></body></html>"""
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    open(out_html, "w").write(page)
    print("dashboard written:", out_html)
    return out_html
