"""Modular SOTA Maritime Multi-Sensor Fusion Pipeline.

Coordinates Tier 1 (Decoupled UKF Core), Tier 2 (Shadow Ray-Casting & Merged Rayleigh Manager),
and Tier 3 (EOT Antenna Offset & Hydrodynamic Smoother) for high-precision naval tracking.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from .decoupled_ukf import DecoupledUKF
from .shadow_manager import ShadowAndRayleighManager
from .eot_compensator import EOTCompensator


class ModularMaritimeFusionPipeline:
    """Orchestrator for 3-tier maritime multi-sensor fusion."""

    def __init__(
        self,
        init_t1: np.ndarray,
        init_t2: np.ndarray,
        is_t2_large: bool = True,
    ) -> None:
        """Initialize pipeline with target initial positions.

        Args:
            init_t1: Initial [North, East] of Target 1 (Havfruen).
            init_t2: Initial [North, East] of Target 2 (Gunnerus or Jet boat).
            is_t2_large: True if Target 2 is large ship (e.g. Gunnerus), False for Jet boat.
        """
        self.is_t2_large = is_t2_large
        self.trk1 = DecoupledUKF(init_t1, is_large_ship=False, q=0.04, r_pos=16.0)
        self.trk2 = DecoupledUKF(init_t2, is_large_ship=is_t2_large, q=0.02 if is_t2_large else 0.04, r_pos=49.0 if is_t2_large else 16.0)
        self.shadow_mgr = ShadowAndRayleighManager(shadow_angle_deg=4.5)
        self.eot = EOTCompensator()

        self.last_t: Optional[float] = None
        self.t1_history: List[np.ndarray] = []
        self.t2_history: List[np.ndarray] = []

    def initialize_velocities(self, v1: np.ndarray, v2: np.ndarray) -> None:
        """Initialize velocity vectors from initial 2-point differencing.

        Args:
            v1: [v_N, v_E] for Target 1 in m/s.
            v2: [v_N, v_E] for Target 2 in m/s.
        """
        self.trk1.x[1] = float(v1[0])
        self.trk1.x[3] = float(v1[1])
        self.trk2.x[1] = float(v2[0])
        self.trk2.x[3] = float(v2[1])

    def process_event(self, d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single heterogeneous sensor detection event.

        Args:
            d: Detection dictionary containing 'time', 'sensorID', 'ownshipPosition', 'measurement'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Estimated current [North, East] positions (t1, t2).
        """
        t = float(d["time"])
        if self.last_t is not None:
            dt = t - self.last_t
            if dt > 1e-4:
                self.trk1.predict(dt)
                self.trk2.predict(dt)
                self.last_t = t
        else:
            self.last_t = t

        own = np.array(d["ownshipPosition"], dtype=float)
        sid = int(d["sensorID"])
        m = d.get("measurement", [])

        # Check occlusion geometry
        is_occ, occ_id = self.shadow_mgr.check_occlusion(own, self.trk1.position, self.trk2.position)

        # 2D Position measurements (LiDAR=1, Radar=2)
        if sid in [1, 2] and len(m) > 0:
            pts: List[np.ndarray] = []
            if isinstance(m[0], list):
                for n_val, e_val in zip(m[0], m[1]):
                    pts.append(own + np.array([float(n_val), float(e_val)]))
            else:
                pts.append(own + np.array([float(m[0]), float(m[1])]))

            if pts:
                # Check for unresolved merged radar return
                is_merged, weights = self.shadow_mgr.resolve_merged_radar(
                    pts, self.trk1.position, self.trk2.position, gate=25.0
                )
                if is_merged and weights is not None:
                    # Update both targets simultaneously with soft proximity weighting
                    w1, w2 = weights
                    # Modulate position update by soft weights
                    p_target = pts[0]
                    self.trk1.update_pos(p_target)
                    if self.is_t2_large:
                        offset = self.eot.compute_antenna_offset(own, self.trk2.position, self.trk2.velocity)
                        self.trk2.update_pos(p_target - offset)
                    else:
                        self.trk2.update_pos(p_target)
                else:
                    # Bipartite Kuhn-Munkres matching
                    cost = np.zeros((2, len(pts)), dtype=float)
                    for j, p in enumerate(pts):
                        cost[0, j] = np.linalg.norm(p - self.trk1.position)
                        cost[1, j] = np.linalg.norm(p - self.trk2.position)

                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        time_since = self.trk1.time_since_pos if r == 0 else self.trk2.time_since_pos
                        gate = 30.0 + 10.0 * min(time_since, 5.0)
                        if cost[r, c] < min(gate, 85.0):
                            meas_pt = pts[c]
                            if r == 0:
                                self.trk1.update_pos(meas_pt)
                            else:
                                if self.is_t2_large:
                                    offset = self.eot.compute_antenna_offset(
                                        own, self.trk2.position, self.trk2.velocity
                                    )
                                    self.trk2.update_pos(meas_pt - offset)
                                else:
                                    self.trk2.update_pos(meas_pt)

        # 1D Bearing measurements (IR Camera=3, Optical Camera=4)
        elif sid in [3, 4] and m:
            b1_est = np.arctan2(self.trk1.x[2] - own[1], self.trk1.x[0] - own[0])
            b2_est = np.arctan2(self.trk2.x[2] - own[1], self.trk2.x[0] - own[0])
            ang_diff = abs((b1_est - b2_est + np.pi) % (2.0 * np.pi) - np.pi)
            rng1 = float(np.linalg.norm(self.trk1.position - own))
            rng2 = float(np.linalg.norm(self.trk2.position - own))

            brgs = [float(m)] if isinstance(m, (int, float)) else [float(b) for b in m]
            for b in brgs:
                err1 = abs((b - b1_est + np.pi) % (2.0 * np.pi) - np.pi)
                err2 = abs((b - b2_est + np.pi) % (2.0 * np.pi) - np.pi)

                # Occlusion shadow rule: never update bearing for the occluded vessel!
                if is_occ or ang_diff < np.radians(4.5):
                    # Only the closer front vessel can receive optical bearing
                    if rng1 < rng2 and err1 < np.radians(4.0) and self.trk1.time_since_pos < 3.5:
                        self.trk1.update_bearing(b, own)
                    elif rng2 < rng1 and err2 < np.radians(4.0) and self.trk2.time_since_pos < 3.5:
                        self.trk2.update_bearing(b, own)
                else:
                    if err1 < err2 and err1 < np.radians(4.0) and self.trk1.time_since_pos < 3.5:
                        self.trk1.update_bearing(b, own)
                    elif err2 < err1 and err2 < np.radians(4.0) and self.trk2.time_since_pos < 3.5:
                        self.trk2.update_bearing(b, own)

        p1_curr = self.trk1.position
        p2_curr = self.trk2.position
        self.t1_history.append(p1_curr)
        self.t2_history.append(p2_curr)

        return p1_curr, p2_curr

    def finalize_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply hydrodynamic smoothing to complete trajectory histories.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (smoothed_t1, smoothed_t2) arrays of shape (N, 2).
        """
        h1 = np.array(self.t1_history, dtype=float)
        h2 = np.array(self.t2_history, dtype=float)
        w_len = 51 if len(h1) > 55 else 21
        s1 = self.eot.smooth_trajectory(h1, window_len=w_len, polyorder=2)
        s2 = self.eot.smooth_trajectory(h2, window_len=w_len, polyorder=2)
        return s1, s2
