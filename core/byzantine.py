"""
Byzantine Fault Tolerance & Model Poisoning Defense Engine
Implements:
1. Multi-Krum Aggregation Defense
2. Cosine Distance Anomaly Detector (Consensus Alignment)
3. Coordinate-wise Median / Trimmed-Mean
4. Adversarial Attack Simulator (Sign Flipping, Gaussian Noise, Backdoor Injection)

Enables Tier 2 Smart Contract / Verification layer to reject poisoned submissions
before executing Federated Averaging (FedAvg).
"""

import numpy as np
from typing import Dict, List, Tuple, Any

class ByzantineDefenseEngine:
    def __init__(self, method: str = "cosine_shield", contamination_factor: float = 0.25):
        """
        Parameters:
        - method: 'cosine_shield', 'multi_krum', or 'trimmed_mean'
        - contamination_factor: expected upper bound ratio of malicious nodes (e.g., 0.25 - 0.33)
        """
        self.method = method
        self.contamination_factor = contamination_factor

    def filter_and_aggregate(
        self,
        client_updates: Dict[str, np.ndarray],
        sample_weights: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Any]]:
        """
        Filters out adversarial/outlier updates and executes robust aggregation.
        Returns:
        - aggregated_weights: np.ndarray
        - accepted_clients: List[str]
        - rejected_clients: List[str]
        - metrics: Dict[str, Any] (anomaly scores, distances, etc.)
        """
        client_ids = list(client_updates.keys())
        n = len(client_ids)

        if n <= 1:
            cid = client_ids[0]
            return client_updates[cid], [cid], [], {"scores": {cid: 0.0}}

        vectors = np.array([client_updates[cid] for cid in client_ids])

        if self.method == "cosine_shield":
            return self._cosine_similarity_filter(client_ids, vectors, client_updates, sample_weights)
        elif self.method == "multi_krum":
            return self._multi_krum_filter(client_ids, vectors, client_updates, sample_weights)
        else: # trimmed_mean / robust median
            return self._trimmed_mean_aggregation(client_ids, vectors, sample_weights)

    def _cosine_similarity_filter(
        self,
        client_ids: List[str],
        vectors: np.ndarray,
        client_updates: Dict[str, np.ndarray],
        sample_weights: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Any]]:
        """
        Calculates pairwise cosine similarity with the median coordinate direction.
        Clients with negative or anomalous cosine alignment (< 0.15) are classified as Byzantine.
        """
        # Direction vector from coordinate-wise median
        median_vector = np.median(vectors, axis=0)
        norm_median = np.linalg.norm(median_vector) + 1e-8

        accepted = []
        rejected = []
        scores = {}

        for cid, vec in zip(client_ids, vectors):
            norm_vec = np.linalg.norm(vec) + 1e-8
            cos_sim = float(np.dot(vec, median_vector) / (norm_vec * norm_median))
            scores[cid] = round(cos_sim, 4)

            # Cosine similarity threshold: normal collaborative updates align positively (> 0.20)
            # Poisoned updates (sign-flip, random noise) have negative or near-zero similarity
            if cos_sim >= 0.20:
                accepted.append(cid)
            else:
                rejected.append(cid)

        # Fallback: if all filtered (e.g. in extreme ties), retain the top scoring client
        if not accepted:
            best_cid = max(scores, key=scores.get)
            accepted.append(best_cid)
            if best_cid in rejected:
                rejected.remove(best_cid)

        # Execute weighted FedAvg on accepted clients
        total_samples = sum(sample_weights.get(cid, 1) for cid in accepted)
        aggregated = np.zeros_like(vectors[0])
        for cid in accepted:
            w = sample_weights.get(cid, 1) / max(1, total_samples)
            aggregated += client_updates[cid] * w

        return aggregated, accepted, rejected, {
            "method": "Cosine Distance Shield",
            "scores": scores,
            "threshold": 0.20
        }

    def _multi_krum_filter(
        self,
        client_ids: List[str],
        vectors: np.ndarray,
        client_updates: Dict[str, np.ndarray],
        sample_weights: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Any]]:
        """
        Multi-Krum algorithm: selects m - f benign candidates minimizing Euclidean distance sums.
        """
        n = len(client_ids)
        f = max(1, int(n * self.contamination_factor))
        m = max(1, n - f)

        # Pairwise euclidean distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(vectors[i] - vectors[j]) ** 2
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Krum scores: sum of n - f - 2 smallest distances
        k_neighbors = max(1, n - f - 1)
        scores = {}
        for i in range(n):
            sorted_dists = np.sort(dist_matrix[i])
            # skip dist to self (0.0)
            score = np.sum(sorted_dists[1:k_neighbors + 1])
            scores[client_ids[i]] = float(score)

        # Sort clients by Krum score (lowest is most benign)
        sorted_cids = sorted(client_ids, key=lambda cid: scores[cid])
        accepted = sorted_cids[:m]
        rejected = sorted_cids[m:]

        # Aggregate accepted with FedAvg
        total_samples = sum(sample_weights.get(cid, 1) for cid in accepted)
        aggregated = np.zeros_like(vectors[0])
        for cid in accepted:
            w = sample_weights.get(cid, 1) / max(1, total_samples)
            aggregated += client_updates[cid] * w

        return aggregated, accepted, rejected, {
            "method": "Multi-Krum",
            "scores": {k: round(v, 2) for k, v in scores.items()},
            "accepted_count": len(accepted)
        }

    def _trimmed_mean_aggregation(
        self,
        client_ids: List[str],
        vectors: np.ndarray,
        sample_weights: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Any]]:
        """
        Coordinate-wise trimmed mean / median aggregation.
        """
        aggregated = np.median(vectors, axis=0)
        return aggregated, client_ids, [], {
            "method": "Coordinate-wise Median",
            "scores": {cid: 1.0 for cid in client_ids}
        }

class AdversarialAttackSimulator:
    """
    Simulates malicious participant behaviors for security testing and live demonstration:
    1. Sign Flipping (reverses and scales weight updates)
    2. Random Gaussian Noise Injection (scrambles gradients)
    3. Targeted Bias Shift / Backdoor
    """
    @staticmethod
    def apply_attack(
        weights: np.ndarray,
        attack_type: str = "sign_flip",
        intensity: float = 3.0
    ) -> np.ndarray:
        poisoned = weights.copy()

        if attack_type == "sign_flip":
            # Invert direction and magnify to derail convergence
            return -intensity * poisoned

        elif attack_type == "gaussian_noise":
            # Add severe variance gaussian noise
            noise = np.random.randn(*poisoned.shape) * (intensity * 2.0)
            return poisoned + noise

        elif attack_type == "backdoor":
            # Inject extreme weight shift into first and last layers to blind IDS
            poisoned[:500] += intensity * 4.0
            poisoned[-100:] -= intensity * 4.0
            return poisoned

        return poisoned
