# Federated and Scalable Product Rating Prediction on Amazon Reviews

DA231o course project, Indian Institute of Science (IISc).
Sanjaya Kumar Sahu · Sachin deep Singh · Navneet Chandan · Gowri Roy Chowdhury

An end-to-end big-data pipeline that predicts the star rating a user would give a product,
built on the ratings-only files of the **Amazon Review Data (2018)** for **all 29 product
categories (~230 M ratings)** and run on **Google Cloud Dataproc** (Spark 3.5.3).

The same model, a damped user/item **bias baseline with ALS fitted on the residuals**, is
trained two ways, centrally with Spark MLlib and with a **simulated federated averaging
(FedAvg)** trainer written on Spark RDDs, and is then served from a **Kafka + Structured
Streaming** scorer. Every stage is benchmarked for strong and weak scaling.

---

## Modules

| Module | What it does | Code |
|---|---|---|
| **1. ETL** | CSV → Parquet (partitioned by category); clean, deduplicate (latest rating per user–item), iterative user/item filters (≥ 2 / ≥ 5 ratings), per-user temporal 80/20 split, integer ID mappings | `etl/` |
| **2. Centralized model** | Damped bias baseline μ + b_u + b_i, then Spark MLlib ALS (rank 20) on the residuals; RMSE/MAE overall, per category, cold (< 5 train ratings) vs warm users; precision@10 | `centralized/`, `evaluation/` |
| **3. Federated model** | Simulated FedAvg of the same model: users hashed into C clients; item factors stay sharded by item across executors (never on the driver); vectorised NumPy solves; communication counted in bytes. C = 1 is the like-for-like control | `federated/` |
| **4a. Scaling benchmarks** | Strong scaling (fixed 20 M training ratings) and weak scaling (1.25 M per executor) for ETL, centralized and federated training at 2, 4, 8 and 16 executors; one Dataproc job per timed run | `benchmarks/`, `scripts/run_benchmarks.py` |
| **4b. Streaming** | Distributed Kafka producer replays test ratings at a fixed rate; Structured Streaming scorer joins them against cached serving tables; latency, throughput, duplicates and streamed RMSE | `streaming/` |

All modules read one config file (`config.yaml`); the category list, data paths, model and
cluster settings all live there.

```
CSVs (GCS) ──► Module 1: Spark ETL ──► train / test Parquet (GCS)
                                          │
                     ┌────────────────────┼─────────────────────┐
                     ▼                    ▼                     ▼
          Module 2: bias + ALS   Module 3: simulated FedAvg   Module 4a: scaling sweeps
                     │                    │
                     └──── saved models ──┘
                                │
          Kafka producer ──► Kafka (master node) ──► Module 4b: Structured Streaming scorer
```

---

## Results (all 29 categories, 16 executors on n2-standard-4 workers)

**Data:** ~230 M raw ratings → **187.1 M** after deduplication and filtering
(22.5 M users, 4.36 M items) → **139.9 M train / 47.2 M test**. 70 % of users are cold
(< 5 training ratings); 58.7 % of training users rate in more than one category.

**Accuracy (test RMSE):**

| Model | Overall | Cold users | Warm users |
|---|---|---|---|
| Global mean | 1.2782 | 1.4075 | 1.2004 |
| Bias baseline | 1.1627 | 1.2889 | **1.0862** |
| Centralized bias + ALS | 1.1597 | 1.2774 | 1.0889 |
| FedAvg, 16 clients | **1.1575** | **1.2753** | 1.0864 |

**Federated client sweep (same job, 10 rounds):**

| Clients | RMSE | Communication | Training time |
|---|---|---|---|
| Centralized (Spark ALS) | 1.1597 | — | 405 s |
| 1 (control) | 1.1598 | 7.2 GB | 964 s |
| 4 | 1.1587 | 25.0 GB | 1,007 s |
| 8 | 1.1580 | 40.3 GB | 1,077 s |
| 16 | 1.1575 | 59.6 GB | 1,174 s |

The one-client control matches Spark's ALS within 0.0001 RMSE. Communication grows 8.2×
from 1 to 16 clients; federation adds 22 % training time, and most of the gap to Spark is
implementation (Python/NumPy vs JVM), not federation.

**Scaling, 2 → 16 executors (one run per setting):**

| Path | Strong speedup (eff.) | Amdahl serial fraction | Weak-scaling efficiency at 16 |
|---|---|---|---|
| Federated training | 5.92× (0.74) | 0.025 | 1.15 |
| Centralized training | 4.44× (0.56) | 0.064 | 1.14 |
| ETL build | 3.49× (0.44) | 0.097 | 0.72 |

Weak-scaling efficiency above 1 for the models reflects that distinct items grow more slowly
than ratings, so item-side work per executor falls as the data grow.

**Streaming (15,000 replayed test ratings at 50 events/s):** every event scored once
(0 duplicates, 0 unknown IDs), 63 micro-batches, throughput ≈ 51 events/s, median latency
5.48 s (centralized model) / 5.51 s (federated model), p95 8.3 / 8.7 s. A restart from the
checkpoint reprocesses nothing.

Raw results are in [`results/`](results/): `scaling_results.csv`, the speedup, efficiency and
weak-scaling charts, and the two streaming summaries with per-event latencies.

---

## Repository layout

```
config.yaml                 reported configuration (all 29 categories)
config.dev*.yaml            small development configs (Colab / single category)
config.gcp.*.yaml           earlier Dataproc test configs
E231_Project_Run.ipynb      one notebook that runs the whole project on Dataproc
E231_Project_Test.ipynb     the same pipeline on a category set you choose (cheap test runs)
etl/                        Module 1: CSV → Parquet, cleaning, filtering, temporal split
centralized/                Module 2: bias baseline + Spark ALS on residuals, model save/load, serving tables
evaluation/                 metrics (RMSE, MAE, precision@k), cold/warm evaluation, gs:// file helpers
federated/                  Module 3: client partitioning, vectorised local ALS, FedAvg coordinator
benchmarks/                 Module 4a: scaling subsets, timed runs, charts (Amdahl fit, weak-scaling efficiency)
streaming/                  Module 4b: Kafka producer replay, Structured Streaming scorer, summary
scripts/                    command-line entry points for every module (spark-submit / gcloud dataproc)
infra/                      Dataproc cluster launch/delete, Kafka init action (broker on the master)
nbtools/                    notebook helpers (cloud setup, charts, dashboard, skip-if-done manifests)
tests/                      NumPy reference test for the federated trainer
legacy_notebooks/           the per-module notebooks used during development
results/                    final scaling and streaming results
```

---

## Getting started

You need your own Google Cloud project with billing and a Google account for Colab and Drive.
**Read [SETUP.md](SETUP.md) first**: it covers copying the code to Drive, setting your project
ID and bucket (the configs contain only the placeholders `YOUR_GCP_PROJECT_ID` and
`YOUR_BUCKET`), the one-time Cloud setup and quotas, and cost.

1. **Try it cheaply:** open `E231_Project_Test.ipynb` in Colab, pick 2–3 small categories and
   2–4 workers, and run the cells top to bottom.
2. **Full run:** `E231_Project_Run.ipynb` with `config.yaml` (all 29 categories) needs 16 workers
   (~68 vCPUs of quota) and several hours of cluster time.
3. **Always run the last cell** (delete the cluster), including after a failure.

The raw data are the per-category "ratings only" CSVs from
<https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/> (headerless:
item, user, rating, timestamp).

### Running modules from the command line

Each module has a script that takes the config path. On Dataproc, submit with
`gcloud dataproc jobs submit pyspark scripts/<script>.py --py-files=code.zip -- --config=gs://…/config.yaml`,
where `code.zip` packages `etl centralized evaluation federated benchmarks`.

| Script | Purpose |
|---|---|
| `scripts/run_etl.py --step convert\|build\|all` | Module 1 |
| `scripts/run_profile.py` | read-only data profile |
| `scripts/run_centralized.py [--tune]` | Module 2 (or the rank / reg_param grid) |
| `scripts/run_federated.py [--num-clients 1 4 8 16]` | Module 3 (defaults to `federated.num_clients`) |
| `scripts/run_benchmarks.py --config config.yaml --modes etl centralized federated` | Module 4a; run from Colab, submits one Dataproc job per timed run, then collects results and draws the charts |
| `scripts/run_streaming.py --action scorer\|producer\|summarize` | Module 4b; the cluster needs Kafka (`KAFKA=1 bash infra/dataproc_launch.sh <workers>`) |

Cluster: `bash infra/dataproc_launch.sh <workers>` and `bash infra/dataproc_delete.sh`
(`CONFIG=other.yaml` to use another config).

### Checking the federated trainer

```
python tests/test_federated_reference.py      # prints ALL PASSED
```

It compares the Spark trainer with a self-contained NumPy reference implementation on
synthetic data.

---

## Notes and limitations

- **Federation is simulated** on an already-centralized public dataset to measure accuracy and
  communication cost. Clients are random hash buckets of users (IID data) and every client takes
  part in every round; no privacy guarantee is provided or implied.
- **One run per scaling setting** (`scaling.repetitions: 1`), so run-to-run variation is not
  measured.
- **Hyperparameters** (rank 20, reg_param 0.3) were tuned on a development set, not re-tuned on
  the 29-category data; warm users are slightly worse than the bias baseline, which points to
  under-regularisation.
- **Precision@10 is about 10⁻⁶**: the model is trained to predict ratings, not to rank 4.4 M items.

## Data and citation

J. Ni, J. Li, J. McAuley. *Justifying recommendations using distantly-labeled reviews and
fine-grained aspects.* EMNLP-IJCNLP 2019.
Dataset: <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>

## License

[MIT](LICENSE)
