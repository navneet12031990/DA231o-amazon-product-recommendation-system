"""Scaling benchmark orchestrator — runs on the CLIENT (Colab or Cloud Shell, needs gcloud), against a
cluster already started with infra/dataproc_launch.sh (SPEC.md Section 5).

  python scripts/run_benchmarks.py --config config.yaml              # everything
  python scripts/run_benchmarks.py --config config.yaml --dry-run    # print commands only
  python scripts/run_benchmarks.py --config config.yaml --modes centralized --sweeps strong

Order: prepare job (capacity check + subsets) -> one job per (repetition, mode, sweep, executors),
each with fixed executors -> collect job -> download CSV -> plots -> upload PNGs.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGES = ["etl", "centralized", "evaluation", "federated", "benchmarks"]
SHOW = ("[scaling]", "[federated]", "[centralized]", "[etl", "Traceback", "Error", "WARNING")


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def sh(cmd: list, dry: bool, label: str) -> None:
    """Run a command, streaming the useful lines as they appear (not only at the end)."""
    print(f"\n=== {label}\n$ {' '.join(cmd)}", flush=True)
    if dry:
        return
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.lstrip().startswith(SHOW) or "state:" in line:
            print(line, flush=True)
    if proc.wait() != 0:
        sys.exit(f"FAILED ({label}), exit code {proc.returncode}. Full output in the Dataproc console "
                 f"(Dataproc -> Jobs). Remember to delete the cluster if you stop here.")


def build_code_zip() -> str:
    tmp = tempfile.mkdtemp()
    stage = os.path.join(tmp, "stage")
    os.makedirs(stage)
    for pkg in PACKAGES:
        shutil.copytree(os.path.join(ROOT, pkg), os.path.join(stage, pkg),
                        ignore=shutil.ignore_patterns("__pycache__"))
    return shutil.make_archive(os.path.join(tmp, "code"), "zip", stage)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the scaling sweeps on a running Dataproc cluster")
    ap.add_argument("--config", required=True, help="local path of the config (e.g. config.yaml)")
    ap.add_argument("--modes", nargs="+", choices=["centralized", "federated", "etl"])
    ap.add_argument("--sweeps", nargs="+", default=["strong", "weak"], choices=["strong", "weak"])
    ap.add_argument("--executors", nargs="+", type=int,
                    help="run only these executor counts (each must be in scaling.executor_counts)")
    ap.add_argument("--repetitions", type=int, help="override scaling.repetitions for this session")
    ap.add_argument("--skip-prepare", action="store_true", help="subsets already built")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, submit nothing")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    infra, sc = config["infra"], config["scaling"]
    modes = args.modes or (["centralized", "federated"] + (["etl"] if sc["etl_sweep_enabled"] else []))
    if "etl" in modes and not sc["etl_sweep_enabled"]:
        sys.exit("--modes etl requested but scaling.etl_sweep_enabled is false in the config")
    cluster, region = infra["cluster_name"], infra["region"]
    counts = [int(n) for n in sc["executor_counts"]]
    if args.executors:
        bad = [n for n in args.executors if n not in counts]
        if bad:
            sys.exit(f"--executors {bad} not in scaling.executor_counts {counts} (subsets exist only for those)")
        counts = [n for n in counts if n in args.executors]
    reps = int(args.repetitions) if args.repetitions else int(sc["repetitions"])
    cfg_uri = _join(sc["scratch_prefix"], "benchmark_config.yaml")
    code_zip = "code.zip" if args.dry_run else build_code_zip()
    job_script = os.path.join(ROOT, "scripts", "run_scaling_job.py")

    def job(extra: list, props: str, label: str) -> None:
        cmd = ["gcloud", "dataproc", "jobs", "submit", "pyspark", job_script,
               f"--cluster={cluster}", f"--region={region}", f"--py-files={code_zip}"]
        if props:
            cmd.append(f"--properties={props}")
        sh(cmd + ["--", f"--config={cfg_uri}"] + extra, args.dry_run, label)

    total = reps * len(modes) * len(args.sweeps) * len(counts)
    print(f"cluster {cluster} ({region}); modes {modes}; sweeps {args.sweeps}; executors {counts}; "
          f"{reps} repetitions -> {total} timed runs")
    sh(["gcloud", "storage", "cp", args.config, cfg_uri], args.dry_run, "upload config")

    if not args.skip_prepare:
        job(["--action=prepare", "--modes", *modes, "--sweeps", *args.sweeps], "",
            "prepare: capacity check + subsets")

    k = 0
    for rep in range(1, reps + 1):              # repetitions outermost: drift spreads over settings
        for mode in modes:
            for sweep in args.sweeps:
                for n in counts:
                    k += 1
                    props = ",".join([
                        "spark.dynamicAllocation.enabled=false",
                        f"spark.executor.instances={n}",
                        f"spark.executor.cores={infra['executor_cores']}",
                        f"spark.executor.memory={infra['executor_memory']}",
                        f"spark.executor.memoryOverhead={infra['executor_memory_overhead']}",
                    ])
                    job(["--action=run", f"--mode={mode}", f"--sweep={sweep}", f"--executors={n}",
                         f"--repetition={rep}"], props, f"run {k}/{total}: {mode} {sweep} n={n} rep={rep}")

    job(["--action=collect"], "", "collect results")

    local_dir = os.path.join(ROOT, "results")
    csv_uri = _join(config["data"]["results_prefix"], "scaling_results.csv")
    sh(["gcloud", "storage", "cp", csv_uri, os.path.join(local_dir, "scaling_results.csv")],
       args.dry_run, "download CSV")
    if args.dry_run:
        print("\n(dry run: plotting skipped)")
        return
    sys.path.insert(0, ROOT)
    from benchmarks.plot_scaling import plot_scaling_results
    pngs = plot_scaling_results(os.path.join(local_dir, "scaling_results.csv"), local_dir)
    if pngs:
        sh(["gcloud", "storage", "cp", *pngs, config["data"]["results_prefix"].rstrip("/") + "/"],
           False, "upload plots")
    print("\nDone. Delete the cluster now: bash infra/dataproc_delete.sh " + args.config)


if __name__ == "__main__":
    main()
