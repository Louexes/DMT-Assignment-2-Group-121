"""End-to-end orchestrator. Runs the seven phase scripts in order.

Usage:
    python scripts/run_all.py                # run everything from phase 1
    python scripts/run_all.py --from-step 4  # resume from phase 4 (LightGBM)
    python scripts/run_all.py --skip 3       # run all except phase 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASES = {
    1: "01_prepare_data.py",
    2: "02_eda.py",
    3: "03_build_features.py",
    4: "04_train_lightgbm.py",
    5: "05_train_rankformer.py",
    6: "06_bias_analysis.py",
    7: "07_make_submission.py",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-step", type=int, default=1)
    p.add_argument("--to-step", type=int, default=7)
    p.add_argument("--skip", type=int, action="append", default=[])
    args = p.parse_args()

    skip = set(args.skip)
    for step, script in PHASES.items():
        if step < args.from_step or step > args.to_step or step in skip:
            continue
        print(f"\n========== phase {step}: {script} ==========")
        t0 = time.time()
        ret = subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT.parent)
        if ret.returncode != 0:
            print(f"[fail] phase {step} returned {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"[done] phase {step} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
