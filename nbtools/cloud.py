"""Colab-side cloud steps: run commands, Cloud Storage checks, quota checks, cluster lifecycle,
Dataproc job submission with timing. Used by E231_Project_Run.ipynb and E231_Project_Test.ipynb."""
import concurrent.futures
import csv
import glob
import io
import json
import os
import shutil
import subprocess
import time

import yaml

SHOW = ("[etl", "[centralized", "[federated", "[scaling", "[producer", "[scorer", "[profile", "[plot",
        "Traceback", "Error", "ERROR", "WARNING", "state:", "Waiting", "Creating", "Deleting", "Kafka:")
UCSD = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/{}.csv"
PACKAGES = ["etl", "centralized", "evaluation", "federated", "benchmarks", "streaming"]
KAFKA_CONNECTOR = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"
KAFKA_JARS_NEEDED = ("spark-sql-kafka-0-10", "spark-token-provider-kafka-0-10", "kafka-clients", "commons-pool2")
STAGE_TIMES = []          # (stage, seconds) for every submitted job, in order


def run(cmd, show_all: bool = False, check: bool = True, env: dict = None) -> list:
    """Run a shell command, stream the useful lines, return ALL output lines (for parsing)."""
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str), env=env or os.environ.copy(), text=True, bufsize=1,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         executable="/bin/bash" if isinstance(cmd, str) else None)
    lines = []
    for line in p.stdout:
        lines.append(line.rstrip("\n"))
        if show_all or line.lstrip().startswith(SHOW) or "state:" in line:
            print(line.rstrip(), flush=True)
    if p.wait() != 0 and check:
        raise RuntimeError(f"FAILED (exit {p.returncode}); see the lines above. Dataproc job logs: Cloud console > "
                           "Dataproc > Jobs. If a cluster is running and you stop here, run the 'Delete the cluster' cell.")
    return lines


def out(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def gs_exists(uri: str) -> bool:
    return subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True).returncode == 0


def gs_cat(uri: str):
    r = subprocess.run(["gcloud", "storage", "cat", uri], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def gs_json(uri: str):
    txt = gs_cat(uri)
    return json.loads(txt) if txt else None


def join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


# ------------------------------------------------------------------ project working copy
def prepare_workdir(drive_dir: str, work: str, project_id: str, bucket: str) -> str:
    """Copy the Drive project to a working folder and fill in project id + bucket. Returns the
    Drive folder actually used (one nested level is found automatically)."""
    src = drive_dir
    if not os.path.exists(os.path.join(src, "CLAUDE.md")):
        inner = glob.glob(os.path.join(src, "*", "CLAUDE.md"))
        assert inner, f"No CLAUDE.md under {src}. Fix DRIVE_PROJECT_DIR."
        src = os.path.dirname(inner[0])
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", "results", "legacy_notebooks"))
    for path in glob.glob(f"{work}/**/*", recursive=True):
        if path.endswith((".yaml", ".py", ".sh", ".md")) and os.path.isfile(path):
            txt = open(path).read()
            new = (txt.replace("YOUR_GCP_PROJECT_ID", project_id).replace("YOUR_BUCKET", bucket)
                      .replace("da231o-spark", bucket))          # older copies used the team bucket name
            if new != txt:
                open(path, "w").write(new)
    return src


# ------------------------------------------------------------------ quotas
def quota_report(project: str, region: str, cfg: dict, workers: int) -> int:
    """Print every quota a Dataproc cluster of this shape uses; return the largest worker count
    that fits all of them (0 if not even the 2-worker minimum fits)."""
    I = cfg["infra"]
    vcpu = int(I["machine_type"].split("-")[-1])
    disk = int(I["boot_disk_size_gb"])
    ssd = I["boot_disk_type"] in ("pd-balanced", "pd-ssd")
    family = I["machine_type"].split("-")[0].upper()
    reg = json.loads(out(["gcloud", "compute", "regions", "describe", region, f"--project={project}", "--format=json"]))
    prj = json.loads(out(["gcloud", "compute", "project-info", "describe", f"--project={project}", "--format=json"]))
    q = {m["metric"]: (m["limit"], m["usage"]) for m in reg.get("quotas", [])}
    q.update({m["metric"]: (m["limit"], m["usage"]) for m in prj.get("quotas", []) if m["metric"] == "CPUS_ALL_REGIONS"})
    per_node = {"CPUS": vcpu, f"{family}_CPUS": vcpu, "CPUS_ALL_REGIONS": vcpu,
                ("SSD_TOTAL_GB" if ssd else "DISKS_TOTAL_GB"): disk, "IN_USE_ADDRESSES": 1}
    nodes = workers + 1
    best = None
    print(f"{'quota':20s} {'needed':>8s} {'free':>8s}   (cluster = 1 master + {workers} workers, {I['machine_type']}, "
          f"{disk} GB {I['boot_disk_type']})")
    for metric, each in per_node.items():
        if metric not in q:
            print(f"{metric:20s} {nodes * each:8.0f} {'?':>8s}   not reported for this project"); continue
        limit, usage = q[metric]
        free = limit - usage
        fits = int(free // each) - 1                       # nodes that fit, minus the master
        best = fits if best is None else min(best, fits)
        flag = "OK" if nodes * each <= free else "TOO LOW"
        print(f"{metric:20s} {nodes * each:8.0f} {free:8.0f}   {flag}")
    best = max(0, best if best is not None else workers)
    print(f"\nLargest worker count that fits every quota above: {best}"
          + ("" if best >= 2 else "  (Dataproc needs >= 2 workers: request more quota)"))
    return best


# ------------------------------------------------------------------ raw data
def ingest_raw(categories: list, raw_prefix: str, source: str = "ucsd", drive_dir: str = None, parallel: int = 4) -> None:
    """Copy each category CSV into the bucket, several at a time (skips files already there)."""
    def one(cat):
        dst = join(raw_prefix, f"{cat}.csv")
        if gs_exists(dst):
            return f"{cat}: already in bucket"
        if source == "ucsd":
            cmd = f"set -o pipefail; curl -fsSL --retry 3 '{UCSD.format(cat)}' | gcloud storage cp - '{dst}'"
        else:
            cmd = f"gcloud storage cp --no-clobber '{drive_dir}/{cat}.csv' '{dst}'"
        r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{cat}: copy failed: {r.stderr[-500:]}")
        return f"{cat}: copied"
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
        for msg in ex.map(one, categories):
            print(msg, flush=True)
    print(f"raw data ready in {time.time() - t0:.0f}s")


def category_sizes(categories: list) -> dict:
    """Bytes of each UCSD ratings-only CSV, from the server's Content-Length (no download)."""
    sizes = {}
    def head(cat):
        r = subprocess.run(["curl", "-sIL", UCSD.format(cat)], capture_output=True, text=True)
        n = [l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if l.lower().startswith("content-length")]
        return cat, int(n[-1]) if n else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for cat, n in ex.map(head, categories):
            sizes[cat] = n
    return sizes


# ------------------------------------------------------------------ Kafka files in the bucket (one time)
def ensure_kafka_files(cfg: dict) -> None:
    """The cluster has no internet access: Kafka and Spark's Kafka connector come from the bucket
    (same steps as the team's KafkaScoring.ipynb one-time cell). Skips what is already there."""
    st = cfg["streaming"]
    tgz, jars = st["kafka_download_url"], st["spark_kafka_jars"].rstrip("/") + "/"
    if not gs_exists(tgz):
        name = tgz.rsplit("/", 1)[-1]                         # kafka_2.13-3.7.1.tgz
        version = name.split("-")[-1].replace(".tgz", "")
        run(f"curl -fsSL https://archive.apache.org/dist/kafka/{version}/{name} -o /tmp/kafka.tgz && "
            f"gcloud storage cp /tmp/kafka.tgz {tgz}", show_all=True)
    else:
        print("Kafka archive already in bucket:", tgz)
    if not gs_exists(jars + "*.jar"):
        run("pip -q install pyspark==3.5.3", show_all=True)
        code = ("from pyspark.sql import SparkSession; s = SparkSession.builder.master('local[1]')"
                f".config('spark.jars.packages', '{KAFKA_CONNECTOR}').getOrCreate(); s.stop()")
        run(["python", "-c", code])
        run(f"gcloud storage cp ~/.ivy2/jars/*.jar {jars}", show_all=True)
    else:
        print("Kafka connector JARs already in bucket:", jars)


def kafka_jars(cfg: dict) -> str:
    listing = out(["gcloud", "storage", "ls", cfg["streaming"]["spark_kafka_jars"].rstrip("/") + "/*.jar"]).split()
    return ",".join(j for j in listing if any(k in j for k in KAFKA_JARS_NEEDED))


# ------------------------------------------------------------------ cluster + jobs
class Cluster:
    """One Dataproc cluster for the whole session (launched with Kafka so streaming can run on it)."""

    def __init__(self, cfg: dict, config_file: str, cfg_uri: str, workers: int, work: str, kafka: bool = True):
        self.cfg, self.config_file, self.cfg_uri, self.workers, self.work, self.kafka = cfg, config_file, cfg_uri, workers, work, kafka
        I = cfg["infra"]
        self.name, self.region = I["cluster_name"], I["region"]
        self.env = dict(os.environ, CONFIG=config_file, KAFKA="1" if kafka else "0")
        n_exec = workers * int(I["executors_per_node"])
        self.props = ",".join([
            "spark.dynamicAllocation.enabled=false", f"spark.executor.instances={n_exec}",
            f"spark.executor.cores={I['executor_cores']}", f"spark.executor.memory={I['executor_memory']}",
            f"spark.executor.memoryOverhead={I['executor_memory_overhead']}",
            f"spark.driver.memory={I['driver_memory']}", f"spark.driver.maxResultSize={I['driver_max_result_size']}"])
        # producer / summary run beside the long-running scorer with ONE small executor (config infra.small_*)
        self.props_small = ",".join([
            "spark.dynamicAllocation.enabled=false", "spark.executor.instances=1",
            f"spark.executor.cores={I['small_executor_cores']}", f"spark.executor.memory={I['small_executor_memory']}",
            f"spark.executor.memoryOverhead={I['small_executor_memory_overhead']}"])
        self.code_zip = None

    def package(self, extra: list = ()) -> str:
        stage = "/tmp/stage"
        shutil.rmtree(stage, ignore_errors=True)
        for pkg in PACKAGES + list(extra):
            shutil.copytree(os.path.join(self.work, pkg), f"{stage}/{pkg}", ignore=shutil.ignore_patterns("__pycache__"))
        self.code_zip = shutil.make_archive("/tmp/code", "zip", stage)
        run(["gcloud", "storage", "cp", os.path.join(self.work, self.config_file), self.cfg_uri], show_all=True)
        return self.code_zip

    def exists(self) -> bool:
        return subprocess.run(["gcloud", "dataproc", "clusters", "describe", self.name, f"--region={self.region}"],
                              capture_output=True).returncode == 0

    def has_kafka(self) -> bool:
        desc = out(["gcloud", "dataproc", "clusters", "describe", self.name, f"--region={self.region}", "--format=json"])
        return "kafka-url" in desc

    def ensure(self) -> None:
        if self.exists():
            return
        print(f"Creating cluster {self.name}: 1 master + {self.workers} workers"
              + (" with a Kafka broker on the master" if self.kafka else ""), flush=True)
        t0 = time.time()
        run(["bash", os.path.join(self.work, "infra/dataproc_launch.sh"), str(self.workers)], show_all=True, env=self.env)
        STAGE_TIMES.append(("create cluster", time.time() - t0))

    def submit(self, script: str, *args, stage: str = None, small: bool = False, jars: str = None,
               asynchronous: bool = False, check: bool = True):
        """Submit scripts/<script> as a Dataproc job. Blocking jobs return their output lines and
        record their wall time (job start-up included) in STAGE_TIMES; async jobs return the job id."""
        self.ensure()
        cmd = ["gcloud", "dataproc", "jobs", "submit", "pyspark", os.path.join(self.work, script),
               f"--cluster={self.name}", f"--region={self.region}", f"--py-files={self.code_zip}",
               f"--properties={self.props_small if small else self.props}"]
        if jars:
            cmd.append(f"--jars={jars}")
        if asynchronous:
            cmd += ["--async", "--format=value(reference.jobId)"]
        cmd += ["--", f"--config={self.cfg_uri}", *args]
        if asynchronous:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-1500:])
            job = r.stdout.strip().splitlines()[-1]
            print(f"started {script} {' '.join(args)} as job {job}")
            return job
        t0 = time.time()
        lines = run(cmd, check=check)
        STAGE_TIMES.append((stage or script, time.time() - t0))
        return lines

    def job_state(self, job: str) -> str:
        return out(["gcloud", "dataproc", "jobs", "describe", job, f"--region={self.region}",
                    "--format=value(status.state)"]).strip()

    def wait(self, job: str) -> list:
        return run(["gcloud", "dataproc", "jobs", "wait", job, f"--region={self.region}"], check=False)

    def delete(self) -> None:
        if self.exists():
            run(["bash", os.path.join(self.work, "infra/dataproc_delete.sh")], show_all=True, check=False, env=self.env)
        else:
            print(f"No cluster {self.name} in {self.region}: nothing to delete.")


# ------------------------------------------------------------------ skip a stage only if it is really done
# A stage is skipped (SKIP_DONE) only if its outputs exist AND a manifest written when it finished
# carries the same fingerprint of the settings that stage depends on. Changing config.yaml (e.g. the
# categories) therefore reruns the stages it affects instead of silently reusing old outputs.
_ETL = [("seed",), ("data", "categories"), ("data", "min_user_interactions"), ("data", "min_item_interactions"),
        ("data", "train_test_split")]
STAGE_SETTINGS = {
    "etl": _ETL,
    "profile": _ETL + [("evaluation", "cold_user_threshold")],
    "centralized": _ETL + [("als",), ("evaluation",)],
    "federated": _ETL + [("als",), ("federated",), ("evaluation", "cold_user_threshold")],
    "benchmarks": _ETL + [("als",), ("federated",), ("scaling",)],
}


def fingerprint(cfg: dict, stage: str, extra=None) -> str:
    import hashlib
    picked = {}
    for path in STAGE_SETTINGS[stage]:
        node = cfg
        for k in path:
            node = node[k]
        picked[".".join(path)] = node
    if extra is not None:
        picked["notebook_overrides"] = extra
    return hashlib.sha256(json.dumps(picked, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _manifest_uri(results_prefix: str, stage: str) -> str:
    return join(results_prefix, f"_manifest/{stage}.json")


def stage_done(cfg: dict, stage: str, results_prefix: str, outputs: list, skip: bool, extra=None) -> bool:
    if not skip:
        return False
    if not all(gs_exists(o) for o in outputs):
        return False
    m = gs_json(_manifest_uri(results_prefix, stage))
    want = fingerprint(cfg, stage, extra)
    if m is None:
        print(f"[{stage}] outputs exist but have no settings manifest (older run): rerunning to be safe")
        return False
    if m.get("fingerprint") != want:
        print(f"[{stage}] outputs exist but were produced with DIFFERENT settings: rerunning")
        return False
    return True


def mark_done(cfg: dict, stage: str, results_prefix: str, extra=None) -> None:
    body = {"stage": stage, "fingerprint": fingerprint(cfg, stage, extra), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = f"/tmp/manifest_{stage}.json"
    open(tmp, "w").write(json.dumps(body, indent=2))
    run(["gcloud", "storage", "cp", tmp, _manifest_uri(results_prefix, stage)])


# ------------------------------------------------------------------ results to local / Drive
def download_results(results_prefix: str, local: str) -> None:
    """Copy the small result files (and streaming outputs) for charts and the dashboard."""
    os.makedirs(local, exist_ok=True)
    for pattern in ["*.json", "*.csv", "*.png"]:
        subprocess.run(f"gcloud storage cp '{join(results_prefix, pattern)}' '{local}/' 2>/dev/null", shell=True)
    for sub in ["streaming_latency", "streaming/batches"]:
        if gs_exists(join(results_prefix, sub)):
            os.makedirs(os.path.join(local, sub), exist_ok=True)
            subprocess.run(["gcloud", "storage", "rsync", "-r", join(results_prefix, sub), os.path.join(local, sub)],
                           capture_output=True)


def append_runs_log(uri: str, rows: list) -> None:
    """Append rows (dicts) to a CSV in the bucket; used by the test notebook's size-scaling chart."""
    old = gs_cat(uri) or ""
    fields = ["run_tag", "n_categories", "raw_rows", "train_rows", "workers", "stage", "seconds"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    if not old:
        w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})
    tmp = "/tmp/runs_log.csv"
    open(tmp, "w").write(old + buf.getvalue())
    run(["gcloud", "storage", "cp", tmp, uri])


def write_yaml(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
