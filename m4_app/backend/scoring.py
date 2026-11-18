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


def score_prediction(predicted_per_observation: list[int],
                     reference_path: list[int] | None) -> dict:
    if reference_path is None:
        return {"available": False, "reason": "No server-side reference file was provided"}
    predicted = collapse_consecutive(predicted_per_observation)
    actual = collapse_consecutive(reference_path)
    scale = max(len(actual), len(predicted))
    similarity = (1 - levenshtein(predicted, actual) / scale) if scale else 1.0
    covered = len(set(predicted) & set(actual)) / len(set(actual)) if actual else 1.0
    return {
        "available": True,
        "route_accuracy_percent": round(100 * similarity, 2),
        "reference_arc_coverage_percent": round(100 * covered, 2),
        "metric_definition": "100 * (1 - Levenshtein(predicted_route, reference_route) / max(sequence_lengths)); consecutive duplicate arc IDs removed",
        "metric_note": "Sequence similarity, NOT GPS-observation Top-1 accuracy. .route contains traversed arc IDs, not per-observation labels.",
        "predicted_arc_count": len(predicted),
        "reference_arc_count": len(actual),
    }
