"""Module 3 correctness test (CLAUDE.md Rule 13): the distributed FedAvg must match an independent
NumPy implementation of the same algorithm to ~1e-9 RMSE. Self-contained: builds its own small
synthetic train/test set. Run from the project root:

  python tests/test_federated_reference.py          (needs pyspark, numpy, pandas)
"""
import copy
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pyspark.sql import SparkSession, functions as F  # noqa: E402
from federated.fedavg_coordinator import run_federated_training  # noqa: E402

TOL = 1e-6  # observed differences are ~1e-9; anything near this tolerance means a real change


def make_data(path):
    rng = np.random.default_rng(1)
    nu, ni, k = 600, 120, 4
    U, V = rng.normal(size=(nu, k)), rng.normal(size=(ni, k))
    ub, ib = rng.normal(0, .5, nu), rng.normal(0, .5, ni)
    rows = []
    for u in range(nu):
        for t, i in enumerate(rng.choice(ni, int(rng.choice([2, 3, 5, 10, 20])), replace=False)):
            r = float(np.clip(np.round(3.8 + ub[u] + ib[i] + .4 * U[u] @ V[i] + rng.normal(0, .5)), 1, 5))
            rows.append((u, int(i), r, t))
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "t"])
    last = df.groupby("user_idx")["t"].transform("max")
    return df[df.t < last].drop(columns="t"), df[df.t == last].drop(columns="t")


def reference(tr, te, cl, cfg, epochs, rounds):
    a, fed = cfg["als"], cfg["federated"]
    R, lam, d, seed, std = a["rank"], a["reg_param"], a["baseline_damping"], cfg["seed"], fed["init_std"]
    t = tr.assign(c=tr.user_idx.map(cl))
    mu = t.rating.mean()
    bi = t.groupby("item_idx").rating.apply(lambda r: (r - mu).sum() / (len(r) + d))
    t["bi"] = t.item_idx.map(bi)
    bu = t.groupby("user_idx").apply(lambda g: (g.rating - mu - g.bi).sum() / (len(g) + d))
    t["y"] = t.rating - mu - t.user_idx.map(bu) - t.bi
    V = {i: np.random.default_rng([seed, int(i)]).normal(0, std, R).astype(np.float32).astype(np.float64)
         for i in t.item_idx.unique()}
    sol = lambda X, y: np.linalg.solve(X.T @ X + lam * len(y) * np.eye(R), X.T @ y)  # noqa: E731
    for _ in range(rounds):
        acc = {}
        for c, g in t.groupby("c"):
            lv = {i: V[i] for i in g.item_idx.unique()}
            for _e in range(epochs):
                Uf = {u: sol(np.stack([lv[i] for i in gg.item_idx]), gg.y.values) for u, gg in g.groupby("user_idx")}
                upd = {i: (sol(np.stack([Uf[u] for u in gg.user_idx]), gg.y.values), len(gg))
                       for i, gg in g.groupby("item_idx")}
                lv = {i: v for i, (v, n) in upd.items()}
            for i, (v, n) in upd.items():
                acc.setdefault(i, []).append((v, n))
        V = {i: sum(v * n for v, n in L) / sum(n for v, n in L) for i, L in acc.items()}
    Uf = {u: sol(np.stack([V[i] for i in g.item_idx]), g.y.values) for u, g in t.groupby("user_idx")}
    tt = te[te.item_idx.isin(V)]
    p = np.clip(mu + tt.user_idx.map(bu).values + tt.item_idx.map(bi).values
                + np.array([Uf[u] @ V[i] for u, i in zip(tt.user_idx, tt.item_idx)]),
                a["prediction_min"], a["prediction_max"])
    return float(np.sqrt(((p - tt.rating.values) ** 2).mean()))


def main():
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    tmp = tempfile.mkdtemp()
    try:
        tr, te = make_data(tmp)
        for name, df in (("train", tr), ("test", te)):
            (spark.createDataFrame(df).select(F.col("user_idx").cast("int"), F.col("item_idx").cast("int"),
                                              F.col("rating").cast("float"))
             .write.parquet(os.path.join(tmp, f"{name}.parquet")))
        cfg = {"seed": 42, "data": {"processed_prefix": tmp},
               "als": {"rank": 6, "reg_param": 0.3, "residual_on_bias": True, "baseline_damping": 10,
                       "prediction_min": 1.0, "prediction_max": 5.0},
               "federated": {"communication_rounds": 2, "local_epochs": 1, "init_std": 0.2,
                             "evaluate_every_round": False, "partitions_per_core": 2,
                             "rows_per_partition": 2000},
               "evaluation": {"cold_user_threshold": 5}}
        failures = 0
        for nc, ep in [(1, 1), (3, 1), (3, 2)]:
            c = copy.deepcopy(cfg)
            c["federated"]["local_epochs"] = ep
            got = run_federated_training(c, nc)["final_rmse"]
            ids = spark.createDataFrame(tr[["user_idx"]].drop_duplicates()).select(
                F.col("user_idx").cast("int").alias("user_idx"))
            cl = dict(ids.select("user_idx", F.pmod(F.xxhash64("user_idx", F.lit(c["seed"])), F.lit(nc))
                                 .cast("int")).toPandas().values)
            exp = reference(tr, te, cl, c, ep, c["federated"]["communication_rounds"])
            ok = abs(got - exp) < TOL
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'}  clients={nc} epochs={ep}: spark {got:.10f}  numpy {exp:.10f}  "
                  f"diff {abs(got - exp):.1e}", flush=True)
        print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
        sys.exit(1 if failures else 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
