"""
smoke_both.py
=============
Sequential smoke test that trains `exp1_low` on TinyLlama-1.1B (the
chosen base model for the DITTP thesis) and prints a single-row summary.

No shell scripts: this is the entire entry point. The user invokes it
from the repo root:

    python code/smoke_both.py

Each phase:
  * Spawns `train.py` as a subprocess and tees its stdout to
    ./outputs/smoke_{shortname}/train.log (live, so crashes are visible).
  * Waits for completion. Returns exit code 0 on success.
  * Parses outputs/exp1_low/trainer_state.json for the loss history
    (the HF Trainer writes that file automatically at end-of-training,
    independent of save_strategy).
  * Best-effort VRAM parse of the captured log: looks for "GiB" / "MiB"
    / "GB" patterns and takes the max. Returns "N/A" if none found --
    the loss and status columns are the source of truth either way.

After the phase a small summary table is printed and the script
exits 0 if the run passed, non-zero otherwise.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")
CONFIG_PATH = os.path.join(CODE_DIR, "config.yaml")

# Model under test. Tuple is (full_model_id, shortname_for_paths).
# Chosen for DITTP thesis: low enough capability ceiling to show
# DITTP-driven deltas, high enough to learn no_robots instructions.
MODELS: list[tuple[str, str]] = [
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "1.1B"),
]
EXPERIMENT = "exp1_low"


# ---------------------------------------------------------------------------
# Subprocess + log capture
# ---------------------------------------------------------------------------

def run_train(model_id: str, shortname: str) -> tuple[int, str]:
    """Spawn train.py for one model and return (returncode, log_path)."""
    log_dir = os.path.join(REPO_ROOT, "outputs", f"smoke_{shortname}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    cmd = [
        sys.executable,
        os.path.join(CODE_DIR, "train.py"),
        "--config", CONFIG_PATH,
        "--model", model_id,
        "--experiment", EXPERIMENT,
        "--smoke",
    ]
    print(f"\n[smoke] $ {' '.join(cmd)}")
    print(f"[smoke] logging to {log_path}")

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.wait()

    return int(proc.returncode), log_path


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def parse_trainer_state(model_id: str) -> dict[str, Any]:
    """Pull the loss history out of outputs/{experiment}/trainer_state.json.

    HF Trainer writes this file at end-of-training regardless of
    save_strategy, so smoke runs (save_strategy='no') still produce it.
    The Trainer writes to `output_dir` (relative to cwd when train.py was
    invoked, which is the repo root), not under code/.
    """
    state_path = os.path.join(REPO_ROOT, "outputs", EXPERIMENT, "trainer_state.json")
    if not os.path.exists(state_path):
        return {"losses": [], "final_loss": None}
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"losses": [], "final_loss": None}

    log_history = state.get("log_history", [])
    losses = [
        float(entry["loss"])
        for entry in log_history
        if "loss" in entry and entry.get("loss") is not None
    ]
    final_loss = losses[-1] if losses else None
    return {"losses": losses, "final_loss": final_loss}


# Matches a float followed by a memory unit, e.g. "18.23 GiB" or "4096 MiB".
_VRAM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(GiB|MiB|GB|MB)\b",
    flags=re.IGNORECASE,
)

_UNIT_TO_GB = {
    "gib": 1.0,        # GiB and GB are close enough for reporting purposes
    "gb": 1.0,
    "mib": 1.0 / 1024.0,
    "mb": 1.0 / 1024.0,
}


def parse_peak_vram(log_path: str) -> str:
    """Best-effort peak VRAM (GB) parsed from the captured stdout log."""
    if not os.path.exists(log_path):
        return "N/A"
    peak_gb = 0.0
    found = False
    with open(log_path) as f:
        for line in f:
            for match in _VRAM_RE.finditer(line):
                value = float(match.group(1))
                unit = match.group(2).lower()
                gb = value * _UNIT_TO_GB[unit]
                if gb > peak_gb:
                    peak_gb = gb
                    found = True
    return f"{peak_gb:.2f} GB" if found else "N/A"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_phase_summary(
    shortname: str,
    returncode: int,
    parsed: dict[str, Any],
    log_path: str,
) -> None:
    status = "PASS" if returncode == 0 else "FAIL"
    final = parsed.get("final_loss")
    losses = parsed.get("losses", [])
    tail = losses[-5:] if losses else []
    vram = parse_peak_vram(log_path)
    print(f"\n=== {shortname} smoke test ===")
    print(f"Final loss: {final:.4f}" if final is not None else "Final loss: N/A")
    print(f"Loss trajectory: {tail}")
    print(f"Peak VRAM: {vram}")
    print(f"Status: {status}")


def print_comparison(rows: list[dict[str, Any]]) -> None:
    """Print a fixed-width side-by-side comparison table."""
    headers = ("Model", "Status", "Final Loss", "Loss Steps", "Peak VRAM")
    str_rows = [
        (
            r["shortname"],
            r["status"],
            f"{r['final_loss']:.4f}" if r["final_loss"] is not None else "N/A",
            str(len(r["losses"])),
            r["peak_vram"],
        )
        for r in rows
    ]
    widths = [max(len(h), max(len(r[i]) for r in str_rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    print("\n=== Smoke test comparison ===")
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for r in str_rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[smoke] config: {CONFIG_PATH}")
    print(f"[smoke] experiment: {EXPERIMENT}")
    print(f"[smoke] models: {[m for m, _ in MODELS]}")

    summary_rows: list[dict[str, Any]] = []
    any_failed = False

    for model_id, shortname in MODELS:
        returncode, log_path = run_train(model_id, shortname)
        parsed = parse_trainer_state(model_id)
        print_phase_summary(shortname, returncode, parsed, log_path)
        if returncode != 0:
            any_failed = True
        summary_rows.append(
            {
                "model_id": model_id,
                "shortname": shortname,
                "status": "PASS" if returncode == 0 else "FAIL",
                "final_loss": parsed.get("final_loss"),
                "losses": parsed.get("losses", []),
                "peak_vram": parse_peak_vram(log_path),
                "returncode": returncode,
            }
        )

    print_comparison(summary_rows)

    if any_failed:
        print("\n[smoke] one or more runs FAILED")
        return 1
    print("\n[smoke] all runs PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
