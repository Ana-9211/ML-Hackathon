"""Reading the raw PathSnap files.

File formats (checked against the actual data, see eda.py):
  .track  lon  lat  t_seconds      (whitespace separated, no header)
  .nodes  lon  lat                 (row index = node_id)
  .arcs   from_node  to_node       (row index = arc_id, directed)
  .route  arc_id                   (chronological reference route)

The reference route is loaded by a *separate* function on purpose: the
prediction pipeline only ever needs `load_trip`, so labels can be kept away
from anything that runs at inference time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "data"))

PARTITIONS = ("train", "student_test", "evaluator_test")


def _read_table(path: Path, dtype) -> np.ndarray:
    # sep=r"\s+" uses pandas' fast C parser for whitespace-delimited files
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c")
    return df.to_numpy()


def partition_dir(partition: str) -> Path:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}, expected one of {PARTITIONS}")
    return DATA_DIR / partition


def list_trips(partition: str) -> list[str]:
    d = partition_dir(partition)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def _trip_file(partition: str, trip_id: str, ext: str) -> Path:
    return partition_dir(partition) / trip_id / f"{trip_id}.{ext}"


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray          # (N, 3) lon, lat, t  -- after cleaning
    nodes: np.ndarray          # (n_nodes, 2) lon, lat
    arcs: np.ndarray           # (n_arcs, 2) from_node, to_node
    cleaning: dict = field(default_factory=dict)

    @property
    def lon(self):
        return self.track[:, 0]

    @property
    def lat(self):
        return self.track[:, 1]

    @property
    def t(self):
        return self.track[:, 2]


def clean_track(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove rows that cannot be used and report what was removed.

    - non-finite values
    - coordinates outside the valid lon/lat range
    - timestamps that are not strictly increasing (duplicates / going back)
    """
    report = {"raw_points": int(len(raw))}
    finite = np.isfinite(raw).all(axis=1)
    valid = finite & (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    report["dropped_invalid"] = int((~valid).sum())
    tr = raw[valid]

    keep = np.ones(len(tr), dtype=bool)
    last_t = -np.inf
    for i, ti in enumerate(tr[:, 2]):          # keep first sample of every timestamp
        if ti <= last_t:
            keep[i] = False
        else:
            last_t = ti
    report["dropped_non_increasing_time"] = int((~keep).sum())
    tr = tr[keep]
    report["clean_points"] = int(len(tr))
    return tr, report


def load_trip(partition: str, trip_id: str) -> Trip:
    """Load GPS track + road network for one trip (no reference route)."""
    raw = _read_table(_trip_file(partition, trip_id, "track"), np.float64)
    track, report = clean_track(raw)
    nodes = _read_table(_trip_file(partition, trip_id, "nodes"), np.float64)
    arcs = _read_table(_trip_file(partition, trip_id, "arcs"), np.int64)
    return Trip(trip_id, partition, track, nodes, arcs, report)


def load_route(partition: str, trip_id: str) -> np.ndarray | None:
    """Reference route (arc ids in travel order). Only for training / scoring."""
    p = _trip_file(partition, trip_id, "route")
    if not p.exists():
        return None
    return _read_table(p, np.int64).ravel()
