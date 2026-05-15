"""Phase 2 — exploratory data analysis on the labeled set."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import eda
from src.data_io import load_labeled


def main() -> None:
    t0 = time.time()
    df = load_labeled()
    print(f"[load] {len(df):,} rows in {time.time() - t0:.1f}s")
    stats = eda.run_all(df)
    out = eda.write_summary(stats)
    print(f"[ok] EDA complete in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
