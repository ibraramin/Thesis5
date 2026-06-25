"""
run_experiments.py
==================
Sequential orchestrator for the six DITTP experiments.

For each experiment listed in config.yaml we:
  1. Skip if outputs/{name}/final already exists (resumability).
  2. Otherwise call train.py via subprocess and wait for it to finish.
  3. Call eval.py against the freshly trained adapter.
  4. Append the eval JSON into a top-level results.json.

No shell scripts. The Python subprocess module does the heavy lifting
so we can capture stdout, set a per-experiment timeout, and keep the
overall loop self-contained.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")
RESULTS_PATH = os.path.join(REPO_ROOT, "results.json")


def run_subprocess(cmd: list[str], log_path: str, timeout: int | None = None) -> int:
    """Run a subprocess, tee stdout to log_path, return the exit code."""
    print(f"[orch] $ {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[orch] TIMEOUT after {timeout}s")
            return 124
    return int(proc.returncode)


def experiment_done(name: str) -> bool:
    final = os.path.join(REPO_ROOT, "outputs", name, "final")
    return os.path.isdir(final) and any(
        fn.startswith("adapter_model") for fn in os.listdir(final)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(CODE_DIR, "config.yaml"))
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to a subset of experiment names")
    parser.add_argument("--smoke", action="store_true", help="Forward --smoke to train.py")
    parser.add_argument("--skip-eval", action="store_true", help="Train only, no eval")
    parser.add_argument("--train-timeout", type=int, default=None, help="Seconds per train run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    experiments: list[dict[str, Any]] = cfg.get("experiments", [])
    if args.only:
        wanted = set(args.only)
        experiments = [e for e in experiments if e["name"] in wanted]

    # Load any pre-existing results so we can append rather than clobber.
    if os.path.exists(args.results):
        with open(args.results) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    for exp in experiments:
        name = exp["name"]
        print("\n" + "=" * 78)
        print(f"[orch] Experiment: {name}")
        print("=" * 78)

        out_dir = os.path.join(REPO_ROOT, "outputs", name)
        os.makedirs(out_dir, exist_ok=True)
        train_log = os.path.join(out_dir, "train.log")
        eval_log = os.path.join(out_dir, "eval.log")

        # ---- training ---------------------------------------------------
        if experiment_done(name):
            print(f"[orch] {name} already trained, skipping train step")
        else:
            train_cmd = [
                sys.executable,
                os.path.join(CODE_DIR, "train.py"),
                "--config", args.config,
                "--experiment", name,
            ]
            if args.smoke:
                train_cmd.append("--smoke")
            # Auto-resume from latest periodic checkpoint if one exists.
            # HF Trainer writes checkpoint-XXX/ dirs on every save_steps.
            ckpt_candidates = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")))
            if ckpt_candidates:
                resume_ckpt = ckpt_candidates[-1]
                print(f"[orch] {name} has checkpoint {resume_ckpt}, will resume from it")
                train_cmd.extend(["--resume-from-checkpoint", resume_ckpt])
            rc = run_subprocess(train_cmd, train_log, timeout=args.train_timeout)
            if rc != 0:
                print(f"[orch] Training failed for {name} (rc={rc}), moving on")
                results[name] = {"status": "train_failed", "rc": rc}
                continue

        # ---- evaluation -------------------------------------------------
        if args.skip_eval:
            results[name] = {"status": "trained_only"}
        else:
            adapter_path = os.path.join(out_dir, "final")
            eval_json = os.path.join(out_dir, "eval_results.json")
            if os.path.exists(eval_json):
                print(f"[orch] {name} already evaluated, loading cached eval_results.json")
                with open(eval_json) as f:
                    results[name] = json.load(f)
                continue
            eval_cmd = [
                sys.executable,
                os.path.join(CODE_DIR, "eval.py"),
                "--adapter", adapter_path,
                "--base", cfg["model"],
                "--out", eval_json,
            ]
            rc = run_subprocess(eval_cmd, eval_log)
            if rc != 0:
                print(f"[orch] Eval failed for {name} (rc={rc})")
                results[name] = {"status": "eval_failed", "rc": rc}
                continue
            with open(eval_json) as f:
                results[name] = json.load(f)

        # Persist after every experiment so a crash doesn't lose progress.
        with open(args.results, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[orch] Wrote aggregated results to {args.results}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
