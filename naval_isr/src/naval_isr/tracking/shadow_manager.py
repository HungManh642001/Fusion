"""Sensor Shadow Ray-Casting & Merged Rayleigh Hit Manager.

Detects Line-of-Sight (LoS) sensor occlusions and resolves merged radar returns
arising from angular Rayleigh resolution limits in littoral environments.
"""

from typing import List, Optional, Tuple
import numpy as np


class ShadowAndRayleighManager:
    """Manages sensor shadowing, occlusion gates, and unresolved merged radar hits."""

    def __init__(
        self,
        shadow_angle_deg: float = 4.5,
        min_unresolved_dist: float = 25.0,
        sigma_r: float = 12.0,
    ) -> None:
        """Initialize the shadow and Rayleigh manager.

        Args:
            shadow_angle_deg: Angular threshold (degrees) below which targets are aligned.
            min_unresolved_dist: Maximum physical separation (meters) for unresolved radar hits.
            sigma_r: Characteristic range spread parameter for soft-weighting.
        """
        self.shadow_angle_rad = np.radians(shadow_angle_deg)
        self.min_unresolved_dist = min_unresolved_dist
        self.sigma_r = sigma_r

    def check_occlusion(
        self, own_pos: np.ndarray, pos1: np.ndarray, pos2: np.ndarray
    ) -> Tuple[bool, int]:
        """Check if one target is occluded in the sensor shadow of the other.

        Args:
            own_pos: Current ownship position [North, East].
            pos1: Target 1 position [North, East].
            pos2: Target 2 position [North, East].

        Returns:
            Tuple[bool, int]: (is_occluded, occluded_target_id) where target_id is 1, 2, or 0.
        """
        r1_sq = (pos1[0] - own_pos[0]) ** 2 + (pos1[1] - own_pos[1]) ** 2
        r2_sq = (pos2[0] - own_pos[0]) ** 2 + (pos2[1] - own_pos[1]) ** 2
        if r1_sq < 4.0 or r2_sq < 4.0:
            return False, 0

        brg1 = np.arctan2(pos1[1] - own_pos[1], pos1[0] - own_pos[0])
        brg2 = np.arctan2(pos2[1] - own_pos[1], pos2[0] - own_pos[0])

        ang_diff = abs((brg1 - brg2 + np.pi) % (2.0 * np.pi) - np.pi)
        if ang_diff < self.shadow_angle_rad:
            # The target farther away from ownship is in the shadow
            if r1_sq < r2_sq:
                return True, 2
            else:
                return True, 1

        return False, 0

    def is_bearing_blocked_by_shadow(
        self,
        z_brg: float,
        own_pos: np.ndarray,
        front_pos: np.ndarray,
        block_margin_deg: float = 5.0,
    ) -> bool:
        """Check if an optical/IR bearing measurement aligns with the front vessel hull.

        Args:
            z_brg: Measured bearing in radians.
            own_pos: Current ownship position [North, East].
            front_pos: Position of the blocking front vessel [North, East].
            block_margin_deg: Angular margin (degrees) considered blocked.

        Returns:
            bool: True if bearing belongs to front vessel and should be blocked for back vessel.
        """
        front_brg = np.arctan2(front_pos[1] - own_pos[1], front_pos[0] - own_pos[0])
        diff = abs((z_brg - front_brg + np.pi) % (2.0 * np.pi) - np.pi)
        return bool(diff < np.radians(block_margin_deg))

    def resolve_merged_radar(
        self,
        pts: List[np.ndarray],
        pos1: np.ndarray,
        pos2: np.ndarray,
        gate: float = 25.0,
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Detect and resolve a single merged radar return from two closely-spaced vessels.

        Args:
            pts: List of 2D detection points from radar [North, East].
            pos1: Target 1 estimated position [North, East].
            pos2: Target 2 estimated position [North, East].
            gate: Association distance gate in meters.

        Returns:
            Tuple[bool, Optional[Tuple[float, float]]]: (is_merged, (w1, w2)) weights summing to 1.0.
        """
        if len(pts) != 1:
            return False, None

        sep = float(np.linalg.norm(pos1 - pos2))
        if sep > max(gate, self.min_unresolved_dist):
            return False, None

        pt = pts[0]
        d1 = float(np.linalg.norm(pt - pos1))
        d2 = float(np.linalg.norm(pt - pos2))

        if min(d1, d2) > gate * 1.6:
            return False, None

        e1 = np.exp(-0.5 * (d1 / self.sigma_r) ** 2)
        e2 = np.exp(-0.5 * (d2 / self.sigma_r) ** 2)
        total = e1 + e2
        if total < 1e-12:
            w1, w2 = 0.5, 0.5
        else:
            w1 = float(e1 / total)
            w2 = float(e2 / total)

        return True, (w1, w2)
