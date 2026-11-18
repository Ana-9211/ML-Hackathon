"""
m3_hmm.transitions
~~~~~~~~~~~~~~~~~~
Road network graph representation and the 3 HMM Transition Probability Models:
  1. Topological Step-Cost Model (Graph connectivity & hop penalties)
  2. Kinematic Exponential Model (Newson & Krumm |d_road - d_gps| decay)
  3. Data-Driven Empirical Transition Matrix (Learned from 80 training trips)
Plus the Soft Log-Linear Ensemble combining all three inside a single Viterbi pass.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd


# Transition relation integer codes between candidate arc a_i (at t-1) and a_j (at t):
REL_SAME_FWD = 0       # Same arc (a_i == a_j) with forward/stationary progress (delta_s >= -jitter)
REL_SAME_BWD = 1       # Same arc (a_i == a_j) moving noticeably backwards (delta_s < -jitter)
REL_HOP1 = 2           # Direct 1-hop connection: arc_to[a_i] == arc_from[a_j] (and not immediate U-turn)
REL_HOP2 = 3           # 2-hop connection: connected via 1 intermediate bridge arc
REL_HOP3 = 4           # 3-hop connection: connected via 2 intermediate bridge arcs (used especially on gaps)
REL_UTURN = 5          # Reverse arc of a two-way street: arc_to[a_i] == arc_from[a_j] and arc_from[a_i] == arc_to[a_j]
REL_DISCONNECTED = 6   # Not connected within 3 hops in the directed road graph


@dataclass
class HMMParams:
    """Tunable parameters for the HMM decoders and the 3-Model Transition Ensemble."""
    # Emission scaling relative to transition log-probabilities
    emission_weight: float = 1.6
    emission_eps: float = 1e-4

    # Ensemble weights over the 3 transition models (w_topo + w_kin + w_emp = 1.0)
    w_topo: float = 0.45
    w_kin: float = 0.35
    w_emp: float = 0.20

    # Technique 1: Topological Graph Step-Cost parameters
    cost_same_fwd: float = 0.0
    cost_same_bwd: float = -2.2
    cost_hop1: float = -0.35
    cost_hop2: float = -1.60
    cost_hop3: float = -2.90
    cost_uturn: float = -4.20
    cost_disconnected: float = -9.50
    backward_jitter_m: float = 5.0

    # Technique 2: Kinematic Distance-Consistency parameters (Newson & Krumm)
    beta_m: float = 18.0          # Exponential scale parameter (meters per second)
    max_kin_penalty: float = 10.0 # Cap on |d_road - d_gps| / (beta * dt)

    # Gap handling thresholds
    gap_reset_sec: float = 30.0   # Above 30s dropout, reset transition penalty (independent sub-trip)
    gap_relax_sec: float = 2.5    # Above 2.5s, relax multi-hop penalties proportionally to dt

    # Route reconstruction shortest-path bridge hops
    max_bridge_hops: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HMMParams":
        p = Path(path)
        if not p.is_file():
            return cls()
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        valid_keys = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


class TripGraph:
    """
    Fast pre-indexed directed road graph for one trip.
    Provides O(1) lookup for 0-hop, 1-hop, 2-hop, and 3-hop shortest connections
    between candidate road segments, plus intermediate bridge arcs for route reconstruction.
    """

    def __init__(self, arc_from: np.ndarray, arc_to: np.ndarray, arc_len: np.ndarray):
        self.arc_from = np.asarray(arc_from, dtype=np.int64)
        self.arc_to = np.asarray(arc_to, dtype=np.int64)
        self.arc_len = np.asarray(arc_len, dtype=np.float64)
        self.n_arcs = len(self.arc_from)
        self.n_nodes = int(max(
            self.arc_from.max() if self.n_arcs else 0,
            self.arc_to.max() if self.n_arcs else 0,
        )) + 1

        # Build outgoing arcs per node: node_id -> list of (arc_id, to_node, length)
        self.out_arcs: dict[int, list[int]] = {}
        for aid in range(self.n_arcs):
            u = int(self.arc_from[aid])
            self.out_arcs.setdefault(u, []).append(aid)

        # Cache for shortest bridge between (end_node_of_a, start_node_of_b)
        self._bridge_cache: dict[tuple[int, int], tuple[int, float, list[int]]] = {}

    def find_bridge(self, node_u: int, node_v: int) -> tuple[int, float, list[int]]:
        """
        Finds the shortest path from node_u (end of arc_i) to node_v (start of arc_j)
        up to 2 intermediate bridge arcs (which corresponds to a 3-hop arc transition).

        Returns:
            (hop_code, bridge_length_m, bridge_arc_ids)
        """
        if node_u == node_v:
            return (REL_HOP1, 0.0, [])

        key = (node_u, node_v)
        cached = self._bridge_cache.get(key)
        if cached is not None:
            return cached

        # Check 1 intermediate arc (REL_HOP2: a_i -> mid_arc -> a_j)
        best_len = float("inf")
        best_mid: list[int] = []
        out_u = self.out_arcs.get(node_u, ())
        for mid_arc in out_u:
            if int(self.arc_to[mid_arc]) == node_v:
                l1 = float(self.arc_len[mid_arc])
                if l1 < best_len:
                    best_len = l1
                    best_mid = [int(mid_arc)]

        if best_mid:
            res = (REL_HOP2, best_len, best_mid)
            self._bridge_cache[key] = res
            return res

        # Check 2 intermediate arcs (REL_HOP3: a_i -> mid1 -> mid2 -> a_j)
        for mid1 in out_u:
            w = int(self.arc_to[mid1])
            l1 = float(self.arc_len[mid1])
            if l1 >= best_len:
                continue
            for mid2 in self.out_arcs.get(w, ()):
                if int(self.arc_to[mid2]) == node_v:
                    total_l = l1 + float(self.arc_len[mid2])
                    if total_l < best_len:
                        best_len = total_l
                        best_mid = [int(mid1), int(mid2)]

        if best_mid:
            res = (REL_HOP3, best_len, best_mid)
        else:
            res = (REL_DISCONNECTED, float("inf"), [])

        self._bridge_cache[key] = res
        return res


class EmpiricalPriors:
    """
    Technique 3: Data-Driven Empirical Transition Priors learned from the 80 training trips.
    Buckets transitions by (dt_bucket, speed_bucket) -> log P(relation_code).
    """

    # dt buckets: [<=1.5s, 1.5..4s, >4s]
    # speed buckets: [<3 m/s (stopped/slow), 3..12 m/s (urban), >12 m/s (fast)]
    def __init__(self, log_table: np.ndarray | None = None):
        # Shape: (3 dt_buckets, 3 speed_buckets, 7 relation codes)
        if log_table is not None:
            self.log_table = np.asarray(log_table, dtype=np.float64)
        else:
            self.log_table = self._default_log_table()

    @staticmethod
    def _default_log_table() -> np.ndarray:
        # Sensible empirical defaults before fitting on training data
        # Columns: [SAME_FWD, SAME_BWD, HOP1, HOP2, HOP3, UTURN, DISCONNECTED]
        probs = np.zeros((3, 3, 7), dtype=np.float64)
        # dt <= 1.5s:
        probs[0, 0] = [0.91, 0.02, 0.06, 0.004, 0.001, 0.004, 0.001]  # slow/stopped
        probs[0, 1] = [0.83, 0.01, 0.14, 0.012, 0.002, 0.004, 0.002]  # urban cruising
        probs[0, 2] = [0.76, 0.005, 0.20, 0.025, 0.005, 0.003, 0.002] # fast road
        # 1.5s < dt <= 4.0s:
        probs[1, 0] = [0.80, 0.02, 0.14, 0.025, 0.005, 0.008, 0.002]
        probs[1, 1] = [0.62, 0.01, 0.28, 0.065, 0.015, 0.007, 0.003]
        probs[1, 2] = [0.48, 0.005, 0.35, 0.120, 0.035, 0.005, 0.005]
        # dt > 4.0s (gaps):
        probs[2, 0] = [0.55, 0.03, 0.24, 0.10, 0.05, 0.015, 0.015]
        probs[2, 1] = [0.32, 0.01, 0.34, 0.18, 0.11, 0.015, 0.025]
        probs[2, 2] = [0.20, 0.01, 0.32, 0.24, 0.17, 0.015, 0.045]
        probs = np.clip(probs, 1e-5, 1.0)
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.log(probs)

    @staticmethod
    def dt_bucket(dt: float) -> int:
        if dt <= 1.5:
            return 0
        if dt <= 4.0:
            return 1
        return 2

    @staticmethod
    def speed_bucket(speed_mps: float) -> int:
        if speed_mps < 3.0:
            return 0
        if speed_mps <= 12.0:
            return 1
        return 2

    @classmethod
    def fit_from_training_df(cls, df_train: pd.DataFrame) -> "EmpiricalPriors":
        """
        Fits the empirical transition probability table directly from the ground-truth
        reference alignments in the training feature table.
        Takes ~0.2 seconds on all 80 training trips.
        """
        # Take one row per (trip_id, point_idx) where label == 1 (or closest candidate to label_arc)
        pos = df_train[df_train["label"] == 1].drop_duplicates(["trip_id", "point_idx"]).sort_values(["trip_id", "point_idx"])
        if len(pos) < 100:
            return cls()

        counts = np.ones((3, 3, 7), dtype=np.float64) * 0.5  # Laplace smoothing

        for _, grp in pos.groupby("trip_id", sort=False):
            t_arr = grp["time"].to_numpy()
            arc_arr = grp["arc_id"].to_numpy()
            u_arr = grp["arc_from"].to_numpy()
            v_arr = grp["arc_to"].to_numpy()
            proj_arr = grp["proj_t"].to_numpy()
            len_arr = grp["arc_len"].to_numpy()
            spd_arr = grp["speed"].to_numpy() if "speed" in grp.columns else np.full(len(grp), 6.0)

            if len(t_arr) < 2:
                continue

            # Build quick node->to_nodes lookup for this trip's observed arcs
            node_next: dict[int, set[int]] = {}
            for u, v in zip(u_arr, v_arr):
                node_next.setdefault(int(u), set()).add(int(v))

            for i in range(1, len(t_arr)):
                dt = float(t_arr[i] - t_arr[i - 1])
                if dt <= 0 or dt >= 30.0:
                    continue
                dtb = cls.dt_bucket(dt)
                spdb = cls.speed_bucket(float(spd_arr[i]))

                a_prev, a_curr = int(arc_arr[i - 1]), int(arc_arr[i])
                u_prev, v_prev = int(u_arr[i - 1]), int(v_arr[i - 1])
                u_curr, v_curr = int(u_arr[i]), int(v_arr[i])

                if a_prev == a_curr:
                    delta_s = (float(proj_arr[i]) - float(proj_arr[i - 1])) * float(len_arr[i])
                    rel = REL_SAME_FWD if delta_s >= -5.0 else REL_SAME_BWD
                elif v_prev == u_curr and u_prev == v_curr:
                    rel = REL_UTURN
                elif v_prev == u_curr:
                    rel = REL_HOP1
                elif u_curr in node_next.get(v_prev, set()):
                    rel = REL_HOP2
                else:
                    # Check 2-step in observed set
                    found3 = False
                    for mid_v in node_next.get(v_prev, set()):
                        if u_curr in node_next.get(mid_v, set()):
                            found3 = True
                            break
                    rel = REL_HOP3 if found3 else REL_DISCONNECTED

                counts[dtb, spdb, rel] += 1.0

        probs = counts / counts.sum(axis=-1, keepdims=True)
        return cls(np.log(np.clip(probs, 1e-6, 1.0)))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.log_table)

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalPriors":
        p = Path(path)
        if p.is_file():
            return cls(np.load(str(p)))
        return cls()


def compute_step_transition_components(
    graph: TripGraph,
    prev_arcs: np.ndarray,
    prev_proj: np.ndarray,
    curr_arcs: np.ndarray,
    curr_proj: np.ndarray,
    dt: float,
    gps_step_m: float,
    speed_mps: float,
    params: HMMParams,
    priors: EmpiricalPriors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the three K_{t-1} x K_t transition log-probability matrices for a single step:
      1. log_A_topo: Topological Graph Step-Cost matrix
      2. log_A_kin:  Kinematic Distance-Consistency matrix (Newson & Krumm)
      3. log_A_emp:  Empirical Data-Driven Transition matrix

    If dt >= params.gap_reset_sec, returns zeros for all three matrices (unconstrained restart after long gap).
    """
    k_prev = len(prev_arcs)
    k_curr = len(curr_arcs)

    if dt >= params.gap_reset_sec or dt <= 0:
        zeros = np.zeros((k_prev, k_curr), dtype=np.float64)
        return zeros, zeros, zeros

    log_topo = np.empty((k_prev, k_curr), dtype=np.float64)
    log_kin = np.empty((k_prev, k_curr), dtype=np.float64)
    log_emp = np.empty((k_prev, k_curr), dtype=np.float64)

    dtb = EmpiricalPriors.dt_bucket(dt)
    spdb = EmpiricalPriors.speed_bucket(speed_mps)
    emp_row = priors.log_table[dtb, spdb]

    # Relax multi-hop topological costs slightly when dt > gap_relax_sec
    gap_factor = 1.0 if dt <= params.gap_relax_sec else max(0.35, params.gap_relax_sec / dt)
    eff_beta = params.beta_m * max(1.0, dt)

    for i in range(k_prev):
        a_i = int(prev_arcs[i])
        u_i = int(graph.arc_from[a_i])
        v_i = int(graph.arc_to[a_i])
        len_i = float(graph.arc_len[a_i])
        p_i = float(prev_proj[i])
        rem_i = (1.0 - p_i) * len_i

        for j in range(k_curr):
            a_j = int(curr_arcs[j])
            u_j = int(graph.arc_from[a_j])
            v_j = int(graph.arc_to[a_j])
            len_j = float(graph.arc_len[a_j])
            p_j = float(curr_proj[j])
            into_j = p_j * len_j

            if a_i == a_j:
                delta_s = (p_j - p_i) * len_i
                if delta_s >= -params.backward_jitter_m:
                    rel = REL_SAME_FWD
                    t_cost = params.cost_same_fwd
                    d_road = max(0.0, delta_s)
                else:
                    rel = REL_SAME_BWD
                    t_cost = params.cost_same_bwd
                    d_road = abs(delta_s) * 1.5
            elif v_i == u_j and u_i == v_j:
                # Immediate U-turn onto reverse arc
                rel = REL_UTURN
                t_cost = params.cost_uturn
                d_road = rem_i + into_j
            elif v_i == u_j:
                # Direct 1-hop connection
                rel = REL_HOP1
                t_cost = params.cost_hop1
                d_road = rem_i + into_j
            else:
                hop_code, bridge_len, _ = graph.find_bridge(v_i, u_j)
                rel = hop_code
                if hop_code == REL_HOP2:
                    t_cost = params.cost_hop2 * gap_factor
                    d_road = rem_i + bridge_len + into_j
                elif hop_code == REL_HOP3:
                    t_cost = params.cost_hop3 * gap_factor
                    d_road = rem_i + bridge_len + into_j
                else:
                    t_cost = params.cost_disconnected
                    d_road = float("inf")

            log_topo[i, j] = t_cost
            log_emp[i, j] = emp_row[rel]

            if np.isfinite(d_road):
                diff = abs(d_road - gps_step_m)
                log_kin[i, j] = -min(params.max_kin_penalty, diff / eff_beta)
            else:
                log_kin[i, j] = -params.max_kin_penalty

    return log_topo, log_kin, log_emp


def compute_transition_matrices(
    graph: TripGraph,
    prev_arcs: np.ndarray,
    prev_proj: np.ndarray,
    curr_arcs: np.ndarray,
    curr_proj: np.ndarray,
    dt: float,
    gps_step_m: float,
    speed_mps: float,
    params: HMMParams,
    priors: EmpiricalPriors,
    mode: str = "ensemble",
) -> np.ndarray:
    """
    Returns the combined transition log-probability matrix for the requested mode:
      - 'topological': Technique 1 only
      - 'kinematic':   Technique 2 only
      - 'empirical':   Technique 3 only
      - 'ensemble':    Weighted average (w_topo * T1 + w_kin * T2 + w_emp * T3)
    """
    log_topo, log_kin, log_emp = compute_step_transition_components(
        graph, prev_arcs, prev_proj, curr_arcs, curr_proj, dt, gps_step_m, speed_mps, params, priors
    )
    if mode == "topological":
        return log_topo
    if mode == "kinematic":
        return log_kin
    if mode == "empirical":
        return log_emp
    # Soft log-linear ensemble
    return (
        params.w_topo * log_topo
        + params.w_kin * log_kin
        + params.w_emp * log_emp
    )
