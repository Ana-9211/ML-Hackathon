"""
m3_hmm.metrics
~~~~~~~~~~~~~~
Pointwise, route-continuity, and sequence-level evaluation metrics for M3.
Matches both M1/M2's pointwise definitions and M4's server-side scoring definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from .transitions import TripGraph


def collapse_consecutive(arcs: list[int] | np.ndarray) -> list[int]:
    """Removes consecutive duplicate arc_ids to produce an ordered route sequence."""
    out: list[int] = []
    for a in arcs:
        val = int(a)
        if not out or out[-1] != val:
            out.append(val)
    return out


def levenshtein_distance(a: list[int], b: list[int]) -> int:
    """Exact Levenshtein edit distance between two ordered arc sequences in O(min(N,M)) memory."""
    if len(b) > len(a):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def compute_route_continuity(route_arcs: list[int] | np.ndarray, graph: TripGraph) -> float:
    """
    Fraction of consecutive transitions in the route sequence that are directly connected
    in the directed road network (arc_to[prev] == arc_from[curr] or prev == curr).
    """
    seq = collapse_consecutive(route_arcs)
    if len(seq) <= 1:
        return 1.0
    connected = 0
    for i in range(len(seq) - 1):
        a_prev = seq[i]
        a_curr = seq[i + 1]
        if a_prev == a_curr or int(graph.arc_to[a_prev]) == int(graph.arc_from[a_curr]):
            connected += 1
    return float(connected / (len(seq) - 1))


def compute_pointwise_continuity(point_arcs: np.ndarray, graph: TripGraph) -> float:
    """
    Fraction of consecutive GPS observations (t-1 -> t) where the predicted road segment
    either stays on the same arc or advances to a directly connected arc.
    """
    if len(point_arcs) <= 1:
        return 1.0
    a_prev = point_arcs[:-1]
    a_curr = point_arcs[1:]
    valid = (a_prev == a_curr) | (graph.arc_to[a_prev] == graph.arc_from[a_curr])
    return float(np.mean(valid))


def evaluate_trip_predictions(
    df_trip: pd.DataFrame,
    pred_point_arcs: np.ndarray,
    pred_route_arcs: list[int],
    graph: TripGraph,
    ref_route: np.ndarray | list[int] | None = None,
) -> dict:
    """
    Computes comprehensive pointwise and sequence-level metrics for a single trip:
      - top1_accuracy_all:      Observation-level Top-1 accuracy across all GPS points
      - top1_accuracy_reliable: Observation-level Top-1 accuracy on reliable points (label_dist <= 50m)
      - candidate_macro_f1:     Candidate-level binary Macro F1-score implied by the chosen arc per step
      - candidate_precision:    Candidate-level binary Precision
      - candidate_recall:       Candidate-level binary Recall
      - pointwise_continuity:   % of consecutive GPS steps with connected/same road segment
      - route_continuity:       % of consecutive deduplicated route segments with arc_to == next_arc_from
      - route_levenshtein_sim:  1 - Levenshtein(pred_route, ref_route) / max(len(pred), len(ref))
      - route_jaccard:          |set(pred_route) & set(ref_route)| / |set(pred_route) | set(ref_route)|
      - route_coverage:         |set(pred_route) & set(ref_route)| / |set(ref_route)|
      - route_f1:               Harmonic mean of route segment precision and recall
    """
    # Map point_idx -> predicted arc_id
    df_pts = df_trip.drop_duplicates("point_idx").sort_values("point_idx")
    point_ids = df_pts["point_idx"].to_numpy(dtype=np.int64)
    label_arcs = df_pts["label_arc"].to_numpy(dtype=np.int64) if "label_arc" in df_pts.columns else None
    reliable = df_pts["label_reliable"].to_numpy(dtype=np.int8) == 1 if "label_reliable" in df_pts.columns else np.ones(len(point_ids), dtype=bool)

    metrics: dict = {
        "n_points": int(len(point_ids)),
        "n_reliable_points": int(reliable.sum()),
        "pointwise_continuity": compute_pointwise_continuity(pred_point_arcs, graph),
        "route_continuity": compute_route_continuity(pred_route_arcs, graph),
        "pred_route_len": int(len(pred_route_arcs)),
    }

    if label_arcs is not None:
        correct_all = pred_point_arcs == label_arcs
        metrics["top1_accuracy_all"] = float(np.mean(correct_all))
        metrics["top1_accuracy_reliable"] = float(np.mean(correct_all[reliable])) if reliable.any() else float(np.mean(correct_all))

        # Candidate-level binary predictions: y_pred = 1 if candidate arc == pred_point_arc[point_idx] else 0
        pt_to_pred = dict(zip(point_ids.tolist(), pred_point_arcs.tolist()))
        cand_pred_arc = df_trip["point_idx"].map(pt_to_pred).to_numpy(dtype=np.int64)
        y_pred_cand = ( df_trip["arc_id"].to_numpy(dtype=np.int64) == cand_pred_arc ).astype(np.int8)
        y_true_cand = df_trip["label"].to_numpy(dtype=np.int8)

        # Evaluate candidate-level Macro F1 on reliable observations (matching M2's convention)
        rel_mask = (df_trip["label_reliable"].to_numpy(dtype=np.int8) == 1) if "label_reliable" in df_trip.columns else np.ones(len(df_trip), dtype=bool)
        yt = y_true_cand[rel_mask] if rel_mask.any() else y_true_cand
        yp = y_pred_cand[rel_mask] if rel_mask.any() else y_pred_cand

        metrics["candidate_macro_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0))
        metrics["candidate_precision"] = float(precision_score(yt, yp, zero_division=0))
        metrics["candidate_recall"] = float(recall_score(yt, yp, zero_division=0))
        metrics["candidate_accuracy"] = float(accuracy_score(yt, yp))

    # Sequence-level route comparison against reference .route
    if ref_route is None and label_arcs is not None:
        ref_seq = collapse_consecutive(label_arcs)
    elif ref_route is not None:
        ref_seq = collapse_consecutive(ref_route)
    else:
        ref_seq = None

    if ref_seq is not None and len(ref_seq) > 0:
        pred_seq = collapse_consecutive(pred_route_arcs)
        scale = max(len(ref_seq), len(pred_seq), 1)
        lev_dist = levenshtein_distance(pred_seq, ref_seq)
        lev_sim = 1.0 - (lev_dist / scale)

        set_pred = set(pred_seq)
        set_ref = set(ref_seq)
        inter = len(set_pred & set_ref)
        union = max(len(set_pred | set_ref), 1)
        cov = inter / max(len(set_ref), 1)
        prec_r = inter / max(len(set_pred), 1)
        f1_r = (2 * prec_r * cov / (prec_r + cov)) if (prec_r + cov) > 0 else 0.0

        metrics["ref_route_len"] = int(len(ref_seq))
        metrics["route_levenshtein_sim"] = float(lev_sim)
        metrics["route_jaccard"] = float(inter / union)
        metrics["route_coverage"] = float(cov)
        metrics["route_f1"] = float(f1_r)

    return metrics
