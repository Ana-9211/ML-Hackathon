"""Server-only evaluation: reference labels are never passed to the predictor."""
from __future__ import annotations
from pathlib import Path
from .dataset import InvalidTrip, find_trip_prefix


def collapse_consecutive(arcs: list[int] | tuple[int, ...]) -> list[int]:
    out: list[int] = []
    for item in arcs:
        if not out or out[-1] != item:
            out.append(item)
    return out


def levenshtein(a: list[int], b: list[int]) -> int:
    # O(min(n,m)) memory, exact ordered-sequence edit distance
    if len(b) > len(a):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def load_reference_private(folder: Path, trip_id: str, arc_count: int) -> list[int] | None:
    """Only scoring calls this method, after inference has finished."""
    prefix = find_trip_prefix(folder, trip_id)
    if prefix is None:
        raise InvalidTrip("The trip has no matching input directory")
    path = Path(str(prefix) + ".route")
    if not path.is_file():
        return None
    result: list[int] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 1:
                raise InvalidTrip(f"Invalid reference route line {line_no}")
            try:
                value = int(parts[0])
            except ValueError as exc:
                raise InvalidTrip(f"Invalid reference arc ID, line {line_no}") from exc
            if not 0 <= value < arc_count:
                raise InvalidTrip(f"Reference arc ID out of range, line {line_no}")
            result.append(value)
    return result


def observation_scores(trip, predicted_per_observation: list[int], reference_path: list[int]) -> dict:
    """Observation-level Top-1 accuracy, as reported in the project report.

    The reference route has no timestamps, so each GPS point is first matched to a
    reference arc with M1's forward-only alignment (m1_data.labels.align_route);
    Top-1 = share of points whose predicted arc equals that aligned arc.
    Runs only after inference, on Laptop 1.
    """
    import numpy as np
    from .pipeline import _ensure_repo_root_importable
    _ensure_repo_root_importable()
    from m1_data.labels import align_route
    from m1_data.loader import Trip
    from m1_data.network import RoadNetwork

    m1_trip = Trip(
        trip_id=trip.trip_id, partition="evaluator_test",
        track=np.array([[o.lon, o.lat, o.timestamp] for o in trip.observations], dtype=np.float64),
        nodes=np.asarray(trip.nodes, dtype=np.float64),
        arcs=np.asarray(trip.arcs, dtype=np.int64),
    )
    net = RoadNetwork(m1_trip)
    px, py = net.proj.to_xy(m1_trip.lon, m1_trip.lat)
    aligned, _, _ = align_route(net, px, py, np.asarray(reference_path, dtype=np.int64))
    pred = np.asarray(predicted_per_observation, dtype=np.int64)
    return {"top1_accuracy_percent": round(100 * float(np.mean(pred == aligned)), 2)}


def score_prediction(predicted_per_observation: list[int],
                     reference_path: list[int] | None, trip=None) -> dict:
    if reference_path is None:
        return {"available": False, "reason": "No server-side reference file was provided"}
    predicted = collapse_consecutive(predicted_per_observation)
    actual = collapse_consecutive(reference_path)
    scale = max(len(actual), len(predicted))
    similarity = (1 - levenshtein(predicted, actual) / scale) if scale else 1.0
    covered = len(set(predicted) & set(actual)) / len(set(actual)) if actual else 1.0
    top1 = {}
    if trip is not None:
        try:
            top1 = observation_scores(trip, predicted_per_observation, reference_path)
        except Exception as exc:  # scoring must never break the demo
            top1 = {"top1_accuracy_percent": None, "top1_note": f"Top-1 unavailable: {exc}"}
    return {
        "available": True,
        **top1,
        "top1_definition": "Share of GPS observations whose predicted arc equals the reference arc aligned to that observation (forward-only alignment, m1_data.labels.align_route)",
        "route_accuracy_percent": round(100 * similarity, 2),
        "reference_arc_coverage_percent": round(100 * covered, 2),
        "metric_definition": "100 * (1 - Levenshtein(predicted_route, reference_route) / max(sequence_lengths)); consecutive duplicate arc IDs removed",
        "metric_note": "route_accuracy_percent is ordered-sequence similarity; top1_accuracy_percent is the observation-level metric.",
        "predicted_arc_count": len(predicted),
        "reference_arc_count": len(actual),
    }
