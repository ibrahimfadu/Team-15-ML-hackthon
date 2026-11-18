"""
src/hmm_decoder.py
Hidden Markov Model (HMM) and Viterbi Decoding for Road Route Reconstruction.

Components:
- Hidden States: Candidate road segments (arcs) at each GPS observation.
- Emission Probabilities: Probabilities output by the trained Random Forest classifier.
- Transition Probabilities: Road network topology adjacency and distance reachability.
- Viterbi Algorithm: Solves the globally optimal, physically unbroken path of arcs.
"""

from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd


class RoadNetworkGraph:
    """
    Lightweight graph representation of road network connectivity from .arcs.
    """
    def __init__(self, arcs_df: pd.DataFrame):
        self.num_arcs = len(arcs_df)
        self.arc_u = arcs_df["u"].values
        self.arc_v = arcs_df["v"].values

        # Build adjacency mapping: node -> list of outgoing arc IDs
        self.outgoing_arcs: Dict[int, List[int]] = {}
        for aid in range(self.num_arcs):
            u = int(self.arc_u[aid])
            if u not in self.outgoing_arcs:
                self.outgoing_arcs[u] = []
            self.outgoing_arcs[u].append(aid)

        # Build direct transition lookup: (arc_i, arc_j) are connected if v_i == u_j
        # We check on the fly using node lookup for memory efficiency

    def is_connected(self, arc_from: int, arc_to: int) -> bool:
        """Returns True if arc_from end node connects to arc_to start node."""
        if arc_from == arc_to:
            return True
        v_from = self.arc_v[arc_from]
        u_to = self.arc_u[arc_to]
        return bool(v_from == u_to)

    def transition_cost(self, arc_from: int, arc_to: int, penalty: float = 1e-5) -> float:
        """
        Computes transition probability P(arc_to | arc_from):
        - Same arc (continuing on road): 0.85
        - Directly connected arc (v_from == u_to): 0.90
        - Disconnected arc: penalty (e.g. 1e-5)
        """
        if arc_from == arc_to:
            return 0.85
        if self.arc_v[arc_from] == self.arc_u[arc_to]:
            return 0.90
        return penalty


class HMMViterbiDecoder:
    """
    Decodes the optimal sequence of road segments across GPS observations.
    """
    def __init__(self, arcs_df: pd.DataFrame, disconnected_penalty: float = 1e-5):
        self.graph = RoadNetworkGraph(arcs_df)
        self.penalty = disconnected_penalty

    def decode(self, candidate_df: pd.DataFrame, emission_probs: np.ndarray) -> List[int]:
        """
        Performs Viterbi decoding in log-probability space.
        candidate_df must have columns ['obs_id', 'candidate_arc_id'].
        emission_probs: 1D array of classifier match probabilities P(match=1) for each candidate row.
        """
        df = candidate_df[["obs_id", "candidate_arc_id"]].copy()
        df["emission"] = np.clip(emission_probs, 1e-6, 1.0 - 1e-6)

        obs_groups = [group for _, group in df.groupby("obs_id", sort=True)]
        num_obs = len(obs_groups)
        if num_obs == 0:
            return []

        # T0: Initial state log-probabilities from first observation
        first_group = obs_groups[0]
        cands_0 = first_group["candidate_arc_id"].values
        viterbi_scores = np.log(first_group["emission"].values)
        backpointers = []

        # Recurrence step: Forward pass
        for t in range(1, num_obs):
            prev_cands = obs_groups[t - 1]["candidate_arc_id"].values
            curr_group = obs_groups[t]
            curr_cands = curr_group["candidate_arc_id"].values
            curr_emissions = curr_group["emission"].values

            num_curr = len(curr_cands)
            new_scores = np.zeros(num_curr)
            curr_backpointers = np.zeros(num_curr, dtype=int)

            for j, c_j in enumerate(curr_cands):
                # Calculate transition from all previous candidates c_i to c_j
                trans_probs = np.array([
                    self.graph.transition_cost(c_i, c_j, self.penalty)
                    for c_i in prev_cands
                ])
                trans_logs = np.log(trans_probs)

                # Total score = previous_score + log(transition) + log(emission)
                total_scores = viterbi_scores + trans_logs + np.log(curr_emissions[j])
                best_prev_idx = int(np.argmax(total_scores))

                new_scores[j] = total_scores[best_prev_idx]
                curr_backpointers[j] = best_prev_idx

            viterbi_scores = new_scores
            backpointers.append(curr_backpointers)

        # Termination: Find best final state
        best_last_idx = int(np.argmax(viterbi_scores))
        decoded_path = [obs_groups[-1]["candidate_arc_id"].values[best_last_idx]]

        # Backtracking
        for t in range(num_obs - 2, -1, -1):
            best_prev_idx = backpointers[t][best_last_idx]
            decoded_path.append(obs_groups[t]["candidate_arc_id"].values[best_prev_idx])
            best_last_idx = best_prev_idx

        decoded_path.reverse()
        return decoded_path

    @staticmethod
    def compress_route(arc_sequence: List[int]) -> List[int]:
        """
        Deduplicates consecutive identical arcs to produce the chronological route sequence.
        e.g. [A, A, A, B, B, C] -> [A, B, C]
        """
        if not arc_sequence:
            return []
        compressed = [arc_sequence[0]]
        for aid in arc_sequence[1:]:
            if aid != compressed[-1]:
                compressed.append(aid)
        return compressed

    def evaluate_route_continuity(self, compressed_route: List[int]) -> float:
        """
        Measures the fraction of transitions in the compressed route that are physically connected.
        A score of 1.0 means 100% physically unbroken route!
        """
        if len(compressed_route) <= 1:
            return 1.0
        connected_count = 0
        total_transitions = len(compressed_route) - 1
        for i in range(total_transitions):
            if self.graph.is_connected(compressed_route[i], compressed_route[i + 1]):
                connected_count += 1
        return float(connected_count / total_transitions)
