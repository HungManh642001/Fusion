"""JDL Level 3 Tactical Threat Assessment & Priority Index (TPI) Engine.

Evaluates maritime threat severity based on:
1. Target kinematic intent (Range to defense asset, Closest Point of Approach CPA, Time to CPA)
2. Target lethal capability (Belief assignment from DSmT/ST-GNN: Warship, Destroyer vs Merchant)
3. Deception indicator (AIS deliberately turned off / Dark vessel status)

Outputs actionable command-and-control countermeasures for decision support.
"""

from typing import Dict, Any, Optional
import numpy as np


class ThreatAssessor:
    """Computes Threat Priority Index (TPI) and tactical countermeasure actions."""

    def __init__(self, coastal_asset_loc: Optional[np.ndarray] = None, max_range_m: float = 50000.0):
        self.asset_loc = coastal_asset_loc if coastal_asset_loc is not None else np.array([0.0, 0.0])
        self.max_range_m = max_range_m

    def compute_threat_priority_index(
        self,
        target_state: Dict[str, Any],
        belief_assignment: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute JDL Level 3 Threat Priority Index (0 to 100).

        Args:
            target_state: Dict with x, y, vx, vy, is_dark
            belief_assignment: Fused belief mass over classes (e.g. Warship, Destroyer, Frigate, Commercial)

        Returns:
            Dict containing tpi_score, threat_level, dcpa_m, tcpa_sec, countermeasure_action
        """
        tx = float(target_state.get("x", 0.0))
        ty = float(target_state.get("y", 0.0))
        tvx = float(target_state.get("vx", 0.0))
        tvy = float(target_state.get("vy", 0.0))
        is_dark = bool(target_state.get("is_dark", False))

        rel_pos = np.array([tx - self.asset_loc[0], ty - self.asset_loc[1]])
        rel_vel = np.array([tvx, tvy])

        dist = float(np.linalg.norm(rel_pos))
        speed = float(np.linalg.norm(rel_vel))

        # 1. Kinematic Intent & Proximity (CPA / TCPA)
        if speed > 1e-3:
            tcpa = -float(np.dot(rel_pos, rel_vel) / (speed**2))
            if tcpa > 0:
                pos_at_cpa = rel_pos + rel_vel * tcpa
                dcpa = float(np.linalg.norm(pos_at_cpa))
            else:
                tcpa = 0.0
                dcpa = dist
        else:
            tcpa = 0.0
            dcpa = dist

        # Range proximity factor [0, 40]
        proximity_factor = max(0.0, 1.0 - (dist / self.max_range_m)) * 40.0

        # Inward velocity factor [0, 20]
        # Closing speed (radial component towards asset)
        if dist > 1.0:
            radial_closing_speed = -float(np.dot(rel_pos, rel_vel) / dist)
        else:
            radial_closing_speed = 0.0
        closing_factor = np.clip(radial_closing_speed / 20.0, 0.0, 1.0) * 20.0

        # 2. Combat Capability Factor [0, 25]
        # Sum mass of military warship classes
        military_belief = (
            belief_assignment.get("Warship", 0.0)
            + belief_assignment.get("Destroyer", 0.0)
            + belief_assignment.get("Frigate", 0.0)
            + belief_assignment.get("Corvette", 0.0)
        )
        capability_factor = military_belief * 25.0

        # 3. Deception / Intent Concealment Factor [0, 15]
        deception_factor = 15.0 if is_dark else 0.0

        # Composite Threat Priority Index (0 to 100)
        tpi_score = float(np.clip(proximity_factor + closing_factor + capability_factor + deception_factor, 0.0, 100.0))

        # Categorize threat level and assign tactical countermeasure
        if tpi_score >= 75.0:
            threat_level = "CRITICAL"
            countermeasure = "DISPATCH_UAV_INTERCEPT_AND_DESIGNATE_TARGET"
        elif tpi_score >= 40.0:
            threat_level = "ELEVATED"
            countermeasure = "ENGAGE_RADAR_FIRE_CONTROL_TRACKING"
        else:
            threat_level = "ROUTINE"
            countermeasure = "MAINTAIN_SURVEILLANCE_AND_LOG"

        return {
            "tpi_score": round(tpi_score, 2),
            "threat_level": threat_level,
            "range_m": round(dist, 1),
            "closing_speed_mps": round(radial_closing_speed, 2),
            "dcpa_m": round(dcpa, 1),
            "tcpa_sec": round(tcpa, 1),
            "military_belief": round(military_belief, 3),
            "is_dark": is_dark,
            "countermeasure_action": countermeasure,
        }
