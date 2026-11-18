"""
m3_hmm
~~~~~~
Sequential route reconstruction using Hidden Markov Models (HMM) and Viterbi decoding.

Includes:
- Three distinct transition probability formulations:
  1. Topological Graph Step-Cost Model
  2. Kinematic Distance-Consistency Exponential Model (Newson & Krumm)
  3. Data-Driven Empirical Transition Matrix Model
- Soft Log-Linear Transition Ensemble combining all three techniques inside a single Viterbi pass
- Output-level Majority Voting across the 3 HMM decoders for ablation comparison
- Post-Viterbi shortest-path bridge reconstruction for connected route sequences
- Parameter tuning (Optuna / Grid search on training trips)
- Comprehensive evaluation (Top-1 Accuracy, Macro F1, Route Continuity, Levenshtein Sequence Similarity, Arc Coverage)
- 5-Layer diagnostic Folium maps and M4 FastAPI live inference adapter
"""

from .transitions import (
    TripGraph,
    HMMParams,
    EmpiricalPriors,
    compute_transition_matrices,
)
from .viterbi import (
    viterbi_decode,
    reconstruct_connected_route,
    decode_trip_all_methods,
)

__all__ = [
    "TripGraph",
    "HMMParams",
    "EmpiricalPriors",
    "compute_transition_matrices",
    "viterbi_decode",
    "reconstruct_connected_route",
    "decode_trip_all_methods",
]
