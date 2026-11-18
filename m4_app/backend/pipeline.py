"""Map-matching inference for M4.

DEMO mode uses a geometrical emission baseline for wiring tests, never a trained ML result.
TRAINED mode requires a trusted joblib M3 classifier or an explicit team predictor adapter.
The matching functions only receive TripInput (no reference routes).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from math import acos, atan2, cos, hypot, log, pi, radians, sin
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree
from .config import Settings
from .dataset import TripInput
from .scoring import collapse_consecutive

EARTH_M = 6371000.0
FEATURE_NAMES = (
    "gps_road_distance_m",
    "bearing_difference_deg",
    "road_segment_length_m",
    "gps_step_distance_m",
    "seconds_since_previous_observation",
    "candidate_distance_rank",
)


class PipelineUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    distance_m: float
    snap_lon: float
    snap_lat: float
    arc_length_m: float
    bearing_diff_deg: float
    rank: int


def _xy(lon: float, lat: float, reference_lat: float) -> tuple[float, float]:
    return (EARTH_M * radians(lon) * cos(radians(reference_lat)), EARTH_M * radians(lat))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _project(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/denom))
    snapped = (a[0] + t*dx, a[1] + t*dy)
    return _dist(p, snapped), snapped


def _angle_difference(p0, p1, a, b) -> float:
    p_dx, p_dy = p1[0] - p0[0], p1[1] - p0[1]
    a_dx, a_dy = b[0] - a[0], b[1] - a[1]
    if hypot(p_dx, p_dy) < 2.0 or hypot(a_dx, a_dy) < 1e-8:
        return 0.0
    dot = (p_dx * a_dx + p_dy * a_dy) / (hypot(p_dx, p_dy) * hypot(a_dx, a_dy))
    return math.degrees(acos(max(-1.0, min(1.0, dot))))


class CandidateIndex:
    """KD-tree preselection over each road segment's midpoint AND endpoints."""
    def __init__(self, trip: TripInput):
        self.trip = trip
        self.ref_lat = sum(lat for _, lat in trip.nodes) / len(trip.nodes)
        self.nodes_xy = [_xy(lon, lat, self.ref_lat) for lon, lat in trip.nodes]
        self.arc_xy = [(self.nodes_xy[a], self.nodes_xy[b]) for a, b in trip.arcs]
        probes = []
        arc_for_probe = []
        for arc_id, (a, b) in enumerate(self.arc_xy):
            probes.extend((a, ((a[0]+b[0])/2, (a[1]+b[1])/2), b))
            arc_for_probe.extend([arc_id] * 3)
        self.probe_arc_ids = arc_for_probe
        self.tree = cKDTree(probes)

    def top_candidates(self, i: int, k: int = 6) -> list[Candidate]:
        obs = self.trip.observations[i]
        p = _xy(obs.lon, obs.lat, self.ref_lat)
        prior = self.trip.observations[i-1] if i else obs
        p0 = _xy(prior.lon, prior.lat, self.ref_lat)
        n_probes = min(len(self.probe_arc_ids), max(80, 12*k))
        _, ids = self.tree.query(p, k=n_probes)
        selected = set(int(self.probe_arc_ids[int(pid)]) for pid in np.atleast_1d(ids))
        values = []
        cos_lat = cos(radians(self.ref_lat))
        for arc_id in selected:
            a, b = self.arc_xy[arc_id]
            distance, snapped = _project(p, a, b)
            snap_lon = math.degrees(snapped[0] / (EARTH_M * cos_lat)) if cos_lat else obs.lon
            snap_lat = math.degrees(snapped[1] / EARTH_M)
            values.append((distance, arc_id, snap_lon, snap_lat, _dist(a, b), _angle_difference(p0, p, a, b)))
        values.sort(key=lambda item: (item[0], item[1]))
        return [Candidate(arc_id, d, lon, lat, length, angle, rank)
                for rank, (d, arc_id, lon, lat, length, angle) in enumerate(values[:k])]


def _make_features(trip: TripInput, candidate_groups: list[list[Candidate]]) -> np.ndarray:
    rows: list[list[float]] = []
    for i, candidates in enumerate(candidate_groups):
        obs = trip.observations[i]
        if i:
            last = trip.observations[i-1]
            x0 = _xy(last.lon, last.lat, obs.lat)
            x1 = _xy(obs.lon, obs.lat, obs.lat)
            movement_m = _dist(x0, x1)
            dt = obs.timestamp - last.timestamp
        else:
            movement_m, dt = 0.0, 0.0
        for cand in candidates:
            rows.append([cand.distance_m, cand.bearing_diff_deg, cand.arc_length_m,
                         movement_m, dt, float(cand.rank)])
    return np.asarray(rows, dtype=float)


@lru_cache(maxsize=2)
def _load_joblib_model(path: str):
    try:
        import joblib
        return joblib.load(path)  # trusted, local team-generated artifacts only
    except Exception as exc:
        raise PipelineUnavailable(f"Cannot load trusted M3 model from {path}: {exc}") from exc


@lru_cache(maxsize=4)
def _load_team_predictor(spec: str) -> Callable[[TripInput], list[int]]:
    try:
        module_name, function_name = spec.rsplit(":", 1)
        predictor = getattr(import_module(module_name), function_name)
        if not callable(predictor):
            raise TypeError("The configured predictor is not callable")
        return predictor
    except Exception as exc:
        raise PipelineUnavailable(f"Could not import M3 predictor adapter: {exc}") from exc


def _emissions(trip: TripInput, candidate_groups: list[list[Candidate]], settings: Settings) -> list[list[float]]:
    if settings.mode == "demo":
        # Synthetic plumbing demo ONLY: not claimed to be an ML classifier.
        return [[-0.5 * (cand.distance_m / 18.0) ** 2 for cand in group] for group in candidate_groups]
    if not settings.model_path.is_file():
        raise PipelineUnavailable("Missing final trained model. Set M4_MODEL_PATH or M4_TEAM_PREDICTOR (see README).")
    model = _load_joblib_model(str(settings.model_path))
    if not hasattr(model, "predict_proba"):
        raise PipelineUnavailable("The M3 model needs predict_proba(), or set M4_TEAM_PREDICTOR to adapt it.")
    features = _make_features(trip, candidate_groups)
    try:
        matrix = np.asarray(model.predict_proba(features))
        classes = list(model.classes_)
        positive_idx = classes.index(1) if 1 in classes else classes.index(True)
        scores = matrix[:, positive_idx]
    except (ValueError, IndexError, AttributeError, TypeError) as exc:
        raise PipelineUnavailable(f"Model/feature incompatibility: {exc}. Check FEATURE_NAMES in pipeline.py.") from exc
    if len(scores) != sum(len(group) for group in candidate_groups):
        raise PipelineUnavailable("The model returned an unexpected number of candidate scores")
    scores = np.clip(scores, 1e-8, 1.0)
    result: list[list[float]] = []
    offset = 0
    for group in candidate_groups:
        result.append([float(log(p)) for p in scores[offset:offset + len(group)]])
        offset += len(group)
    return result


def _transition(trip: TripInput, old_arc: int, new_arc: int, dt: float) -> float:
    if dt >= 30.0:  # independent segment of a trajectory after a large gap
        return 0.0
    if old_arc == new_arc:
        return -0.12
    old_start, old_end = trip.arcs[old_arc]
    new_start, new_end = trip.arcs[new_arc]
    if old_end == new_start:  # directed legal next segment
        return -0.35
    if old_start == new_start or old_end == new_end:  # possible turnaround/intersection
        return -3.0
    return -9.0  # penalise impossible jumps; do not invent connecting road arcs


def viterbi(trip: TripInput, groups: list[list[Candidate]], log_emissions: list[list[float]]) -> list[int]:
    if len(groups) != len(trip.observations) or len(log_emissions) != len(groups):
        raise ValueError("Candidate or emission count does not match GPS observations")
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError("Each GPS observation must have at least one candidate")
    dp = list(log_emissions[0])
    backs: list[list[int]] = [[-1] * len(groups[0])]
    for t in range(1, len(groups)):
        dt = trip.observations[t].timestamp - trip.observations[t-1].timestamp
        next_dp: list[float] = []
        parent_ids: list[int] = []
        for j, cand in enumerate(groups[t]):
            possibilities = [dp[i] + _transition(trip, prev.arc_id, cand.arc_id, dt)
                             for i, prev in enumerate(groups[t-1])]
            parent = max(range(len(possibilities)), key=possibilities.__getitem__)
            next_dp.append(possibilities[parent] + log_emissions[t][j])
            parent_ids.append(parent)
        backs.append(parent_ids)
        dp = next_dp
    state = max(range(len(dp)), key=dp.__getitem__)
    indices = [state]
    for t in range(len(groups)-1, 0, -1):
        state = backs[t][state]
        indices.append(state)
    indices.reverse()
    return [group[idx].arc_id for group, idx in zip(groups, indices)]


def predict_trip(trip: TripInput, settings: Settings) -> tuple[list[int], str]:
    """Returns ONE inferred arc ID per observation; NEVER accepts a reference route."""
    if settings.mode not in ("demo", "trained"):
        raise PipelineUnavailable("M4_MODE must be 'trained' or 'demo'")
    if settings.mode == "trained" and settings.team_predictor:
        predictor = _load_team_predictor(settings.team_predictor)
        try:
            result = list(predictor(trip))
        except Exception as exc:
            raise PipelineUnavailable(f"Team predictor failed: {exc}") from exc
        source = "team_m3_adapter"
    else:
        index = CandidateIndex(trip)
        candidates = [index.top_candidates(i) for i in range(len(trip.observations))]
        scores = _emissions(trip, candidates, settings)
        result = viterbi(trip, candidates, scores)
        source = "geometry_demo_baseline" if settings.mode == "demo" else "joblib_m3_plus_viterbi"
    if len(result) != len(trip.observations):
        raise PipelineUnavailable("Predictor must return exactly one arc ID per GPS observation")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 or x >= len(trip.arcs)
           for x in result):
        raise PipelineUnavailable("Predictor returned an invalid 0-based arc ID")
    return [int(x) for x in result], source


def prediction_geometry(trip: TripInput, predicted_ids: list[int]) -> dict:
    selected = collapse_consecutive(predicted_ids)
    segments = []
    for arc_id in selected:
        i, j = trip.arcs[arc_id]
        a, b = trip.nodes[i], trip.nodes[j]
        segments.append({"arc_id": arc_id, "coordinates": [[a[1], a[0]], [b[1], b[0]]]})
    connected = sum(old == new or trip.arcs[old][1] == trip.arcs[new][0]
                    for old, new in zip(selected, selected[1:]))
    denominator = max(1, len(selected) - 1)
    return {
        "arc_ids_per_observation": predicted_ids,
        "ordered_arcs": selected,
        "segments": segments,
        "continuity_percent": round(100 * connected / denominator, 2) if len(selected) > 1 else 100.0,
    }
