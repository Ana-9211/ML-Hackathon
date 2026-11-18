"""Input-only trip loading. This module intentionally never reads *.route labels."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import re

TRIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


class InvalidTrip(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    lon: float
    lat: float
    timestamp: float


@dataclass(frozen=True)
class TripInput:
    trip_id: str
    observations: tuple[Observation, ...]
    nodes: tuple[tuple[float, float], ...]  # longitude, latitude
    arcs: tuple[tuple[int, int], ...]  # from_node, to_node; arc ID = zero-based row


def safe_trip_id(value: str) -> str:
    if not TRIP_ID.fullmatch(value) or value in (".", "..") or ".." in value:
        raise InvalidTrip("Invalid trip identifier")
    return value


def find_trip_prefix(folder: Path, trip_id: str) -> Path | None:
    """Find flat files or a directory named for the trip with a complete file trio."""
    safe_trip_id(trip_id)
    flat = folder / trip_id
    if all(Path(str(flat) + suffix).is_file() for suffix in (".track", ".nodes", ".arcs")):
        return flat
    child = folder / trip_id
    if not child.is_dir():
        return None
    same_stem = child / trip_id
    if all(Path(str(same_stem) + suffix).is_file() for suffix in (".track", ".nodes", ".arcs")):
        return same_stem
    # Nested datasets sometimes store their one trip with a different basename.
    valid = []
    for track in child.glob("*.track"):
        prefix = track.with_suffix("")
        if all(Path(str(prefix) + suffix).is_file() for suffix in (".nodes", ".arcs")):
            valid.append(prefix)
    return valid[0] if len(valid) == 1 else None


def list_trip_ids(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    names = {p.stem for p in folder.glob("*.track")}
    names.update(p.name for p in folder.iterdir() if p.is_dir())
    return sorted(name for name in names
                  if TRIP_ID.fullmatch(name) and ".." not in name
                  and find_trip_prefix(folder, name) is not None)


def _matrix(path: Path, cols: int) -> list[list[float]]:
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != cols:
                raise InvalidTrip(f"{path.suffix} line {n}: expected {cols} columns")
            try:
                floats = [float(s) for s in parts]
            except ValueError as exc:
                raise InvalidTrip(f"{path.suffix} line {n}: not numeric") from exc
            if not all(math.isfinite(v) for v in floats):
                raise InvalidTrip(f"{path.suffix} line {n}: invalid number")
            values.append(floats)
    return values


def _validate_lonlat(lon: float, lat: float) -> None:
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise InvalidTrip("Longitude/latitude outside valid range")


def _as_index(value: float, max_count: int) -> int:
    if int(value) != value or not (0 <= value < max_count):
        raise InvalidTrip("An arc references an invalid 0-based node index")
    return int(value)


def load_trip(folder: Path, trip_id: str, max_observations: int = 50000,
              max_arcs: int = 1000000) -> TripInput:
    safe_trip_id(trip_id)
    prefix = find_trip_prefix(folder, trip_id)
    if prefix is None:
        raise FileNotFoundError(f"Trip {trip_id!r} was not found")
    track = _matrix(Path(str(prefix) + ".track"), 3)
    nodes = _matrix(Path(str(prefix) + ".nodes"), 2)
    arcs = _matrix(Path(str(prefix) + ".arcs"), 2)
    if not track or not nodes or not arcs:
        raise InvalidTrip("Track, nodes and arcs must not be empty")
    if len(track) > max_observations or len(arcs) > max_arcs:
        raise InvalidTrip("Trip exceeds configured safety limit")
    for lon, lat, _timestamp in track:
        _validate_lonlat(lon, lat)
    for lon, lat in nodes:
        _validate_lonlat(lon, lat)
    for i in range(1, len(track)):
        if track[i][2] < track[i - 1][2]:
            raise InvalidTrip("GPS timestamps are not chronological")
    arc_pairs = tuple((_as_index(a, len(nodes)), _as_index(b, len(nodes))) for a, b in arcs)
    return TripInput(
        trip_id=trip_id,
        observations=tuple(Observation(lon, lat, ts) for lon, lat, ts in track),
        nodes=tuple((lon, lat) for lon, lat in nodes),
        arcs=arc_pairs,
    )
