"""Build the candidate/feature tables for the train and student_test partitions.

Usage (from the repository root):
    python -m m1_data.build_dataset

Outputs:
    m1_data/outputs/features/features_<partition>.parquet   (not committed, regenerate)
    m1_data/outputs/dataset_stats_<partition>.csv           (per-trip summary, committed)

The evaluator_test partition is deliberately NOT processed here: its labels
must never be part of any development table.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .features import build_trip_table
from .loader import list_trips, load_route, load_trip

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FEATURE_DIR = OUT_DIR / "features"
DEV_PARTITIONS = ("train", "student_test")


def build_partition(partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables, stats = [], []
    for k, tid in enumerate(list_trips(partition), 1):
        t0 = time.perf_counter()
        trip = load_trip(partition, tid)
        df, st = build_trip_table(trip, load_route(partition, tid))
        st["build_seconds"] = round(time.perf_counter() - t0, 2)
        tables.append(df)
        stats.append(st)
        print(f"[{partition}] {k:3d} trip {tid}: {st['points']:6d} pts, {st['candidate_rows']:7d} rows, "
              f"miss={st['candidate_miss_rate']:.4f}  ({st['build_seconds']} s)", flush=True)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=list(DEV_PARTITIONS), choices=DEV_PARTITIONS)
    args = ap.parse_args()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for part in args.partitions:
        df, stats = build_partition(part)
        df.to_parquet(FEATURE_DIR / f"features_{part}.parquet", index=False)
        stats.to_csv(OUT_DIR / f"dataset_stats_{part}.csv", index=False)
        print(f"[{part}] {len(stats)} trips, {len(df):,} candidate rows, "
              f"positive rate {df['label'].mean():.3f}, "
              f"point-level candidate miss rate {1 - df.groupby(['trip_id', 'point_idx'])['label'].max().mean():.4f}")


if __name__ == "__main__":
    main()
