# M3 — HMM and Viterbi route reconstruction

Owner: M3

Scope:
- Hidden states, emission and transition model, Viterbi decoding
- Missing candidates, large GPS gaps and disconnected road segments
- Decoder parameter tuning (train + student_test only)
- Classifier-only vs classifier + HMM comparison: point accuracy, sequence-level metrics, route continuity
- Folium maps of predicted vs reference routes

Inputs: the M2 model's probabilities plus M1's candidate table (`arc_from`, `arc_to`, `proj_t`, `time` columns) and `m1_data.network.RoadNetwork` (arc lengths, adjacency).
