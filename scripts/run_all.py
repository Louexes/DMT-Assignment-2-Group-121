"""End-to-end orchestrator. Runs the phase scripts in order.

Usage:
    python scripts/run_all.py                # run everything from phase 1
    python scripts/run_all.py --from-step 4  # resume from phase 4 (LightGBM)
    python scripts/run_all.py --skip 4b      # run all except phase 4b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ordered phase list. Phase 4b trains a second LightGBM booster used as a
# diverse leg of the three-way ensemble in phase 7.
PHASES: list[tuple[str, str]] = [
    ("1", "01_prepare_data.py"),
    ("2", "02_eda.py"),
    ("3", "03_build_features.py"),
    ("4", "04_train_lightgbm.py"),
    ("4b", "04b_train_lightgbm_v3.py"),
    ("5", "05_train_rankformer.py"),
    ("6", "06_bias_analysis.py"),
    ("7", "07_make_submission.py"),
]


def _phase_geq(a: str, b: str) -> bool:
    """Compare phase labels like '4', '4b', '5' lexicographically with
    integer-aware ordering on the leading numeric prefix."""
    def key(p: str) -> tuple[int, str]:
        i = 0
        while i < len(p) and p[i].isdigit():
            i += 1
        return (int(p[:i] or "0"), p[i:])
    return key(a) >= key(b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-step", type=str, default="1")
    p.add_argument("--to-step", type=str, default="7")
    p.add_argument("--skip", type=str, action="append", default=[])
    args = p.parse_args()

    skip = set(args.skip)
    for label, script in PHASES:
        if not _phase_geq(label, args.from_step):
            continue
        if not _phase_geq(args.to_step, label):
            continue
        if label in skip:
            continue
        print(f"\n========== phase {label}: {script} ==========")
        t0 = time.time()
        ret = subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT.parent)
        if ret.returncode != 0:
            print(f"[fail] phase {label} returned {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"[done] phase {label} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
