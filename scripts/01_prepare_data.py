"""Phase 1 — convert the two CSV files to Parquet.

Run from project root:
    python scripts/01_prepare_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.data_io import csv_to_parquet


def main() -> None:
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path, parquet_path, label in [
        (C.LABELED_CSV, C.LABELED_PARQUET, "labeled (test.csv)"),
        (C.SUBMIT_CSV, C.SUBMIT_PARQUET, "submit (train.csv)"),
    ]:
        if parquet_path.exists():
            print(f"[skip] {parquet_path.name} already exists")
            continue
        t0 = time.time()
        rows = csv_to_parquet(csv_path, parquet_path)
        size_mb = parquet_path.stat().st_size / 1024**2
        print(
            f"[ok] {label}: {rows:,} rows -> {parquet_path.name} "
            f"({size_mb:.1f} MB) in {time.time() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()
