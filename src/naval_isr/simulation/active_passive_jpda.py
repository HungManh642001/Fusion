"""Active-Passive Joint Probabilistic Data Association (JPDA) Engine.

Solves the multi-target association problem between:
1. Active 2D/3D Radar kinematic tracks (Range, Azimuth, Doppler)
2. Passive ESM 1D Line-of-Bearing (LOB) measurements (Bearing-only)

Key innovation:
Bearing-only angular Mahalanobis distance gating to prevent track crossover ID swaps
when ships cross at identical bearings from different ranges.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class ActivePassiveJPDA:
    """Active-Passive Radar and ESM Data Association filter."""

    def __init__(
        self,
        gate_threshold: float = 9.0,  # 3-sigma gate for 1-DOF chi2 (chi2_inv(0.99, 1) ~ 6.63 -> 9.0 is safe)
        cartesian_gate_dist_m: float = 300.0,
        sigma_radar_theta: float = 0.5 * np.pi / 180.0,
        sigma_esm_theta: float = 1.0 * np.pi / 180.0,
    ):
        self.gate_threshold = gate_threshold
        self.cartesian_gate_dist_m = cartesian_gate_dist_m
        self.sigma_radar_theta = sigma_radar_theta
        self.sigma_esm_theta = sigma_esm_theta

    @staticmethod
    def wrap_angle_rad(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def compute_angular_gating(
        self,
        radar_track_state: np.ndarray,
        esm_bearing: float,
        sigma_theta: Optional[float] = None,
        sigma_esm: Optional[float] = None,
    ) -> float:
        """Compute angular Mahalanobis distance squared between 2D radar track and 1D ESM bearing."""
        sig_th = sigma_theta if sigma_theta is not None else self.sigma_radar_theta
        sig_esm = sigma_esm if sigma_esm is not None else self.sigma_esm_theta

        # Extract (x, y) from state [x, vx, y, vy]
        tx = float(radar_track_state[0])
        ty = float(radar_track_state[2])

        track_bearing = np.arctan2(ty, tx)
        delta_angle = self.wrap_angle_rad(track_bearing - esm_bearing)

        var_total = sig_th**2 + sig_esm**2
        if var_total < 1e-12:
            var_total = 1e-12

        d2 = (delta_angle**2) / var_total
        return float(d2)

    def associate(
        self,
        tracks: List[Dict[str, Any]],
        radar_meas: List[Dict[str, Any]],
        esm_meas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Associate tracks with incoming radar and passive ESM measurements."""
        results: List[Dict[str, Any]] = []

        assigned_radar_indices = set()
        assigned_esm_indices = set()

        for track in tracks:
            track_id = track["track_id"]
            state = track["state"]
            tx, ty = state[0], state[2]

            # 1. Associate Radar measurement (Cartesian 2D nearest neighbor within gate)
            best_r_idx = None
            min_r_dist = float("inf")

            for idx, r_item in enumerate(radar_meas):
                if idx in assigned_radar_indices:
                    continue
                m_cart = r_item["meas_cartesian"]
                dist = np.hypot(tx - m_cart[0], ty - m_cart[1])
                if dist < self.cartesian_gate_dist_m and dist < min_r_dist:
                    min_r_dist = dist
                    best_r_idx = idx

            associated_radar = None
            if best_r_idx is not None:
                assigned_radar_indices.add(best_r_idx)
                associated_radar = radar_meas[best_r_idx]

            # 2. Associate Passive ESM measurement (Angular Mahalanobis gating)
            best_esm_idx = None
            min_esm_dist2 = float("inf")

            for idx, esm_item in enumerate(esm_meas):
                if idx in assigned_esm_indices:
                    continue
                esm_bearing = esm_item["bearing_rad"]
                d2 = self.compute_angular_gating(state, esm_bearing)
                if d2 <= self.gate_threshold and d2 < min_esm_dist2:
                    min_esm_dist2 = d2
                    best_esm_idx = idx

            associated_esm = None
            if best_esm_idx is not None:
                assigned_esm_indices.add(best_esm_idx)
                associated_esm = esm_meas[best_esm_idx]

            results.append({
                "track_id": track_id,
                "state": state,
                "associated_radar": associated_radar,
                "associated_esm": associated_esm,
                "angular_mahalanobis_d2": min_esm_dist2 if best_esm_idx is not None else None,
            })

        return results
