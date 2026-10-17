"""AIS Dead-Reckoning Extrapolation and Spaceborne SAR Correlation Engine.

Solves the time-asynchrony problem between:
1. Periodic, asynchronous AIS transponder messages
2. Instantaneous snapshot of Spaceborne SAR imagery (e.g. Sentinel-1, xView3)

Features:
- Great-circle geodesic Dead-Reckoning (DR) extrapolation
- Spatial gating via Haversine distance
- 3-way maritime association:
  - Matched (Cooperative / Verified Vessel)
  - Unmatched SAR (Dark Vessel Candidate)
  - Unmatched AIS (Ghost Ship / Spoofed Target Candidate)
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


class AISCorrelationEngine:
    """Correlates instantaneous SAR satellite detections with asynchronous AIS reports."""

    def __init__(self, earth_radius_km: float = 6371.0):
        self.R = earth_radius_km
        self.R_m = earth_radius_km * 1000.0

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two geographic coordinates in kilometers."""
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)

        a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(6371.0 * c)

    def extrapolate_dead_reckoning(self, ais: Dict[str, Any], target_ts: float) -> Dict[str, Any]:
        """Extrapolate AIS coordinates to target_ts using speed (SOG) and course (COG)."""
        ais_ts = float(ais.get("ts", ais.get("timestamp", 0.0)))
        dt_sec = target_ts - ais_ts

        lat1_rad = np.radians(ais["lat"])
        lon1_rad = np.radians(ais["lon"])

        # Speed Over Ground in m/s (1 knot = 0.514444 m/s)
        sog_knots = float(ais.get("sog", ais.get("sog_knots", 0.0)))
        v_mps = sog_knots * 0.514444

        # Course Over Ground: degrees clockwise from true north
        cog_deg = float(ais.get("cog", ais.get("cog_deg", 0.0)))
        theta_rad = np.radians(cog_deg)

        dist_m = v_mps * dt_sec
        angular_dist = dist_m / self.R_m

        # Geodesic direct formula
        lat2_rad = np.arcsin(
            np.sin(lat1_rad) * np.cos(angular_dist)
            + np.cos(lat1_rad) * np.sin(angular_dist) * np.cos(theta_rad)
        )
        lon2_rad = lon1_rad + np.arctan2(
            np.sin(theta_rad) * np.sin(angular_dist) * np.cos(lat1_rad),
            np.cos(angular_dist) - np.sin(lat1_rad) * np.sin(lat2_rad),
        )

        extrapolated = dict(ais)
        extrapolated["lat"] = float(np.degrees(lat2_rad))
        extrapolated["lon"] = float(np.degrees(lon2_rad))
        extrapolated["extrapolated_dt_sec"] = dt_sec
        return extrapolated

    def match_sar_to_ais(
        self,
        sar_detections: List[Dict[str, Any]],
        ais_records: List[Dict[str, Any]],
        target_ts: float,
        gate_dist_km: float = 1.5,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Match SAR detections with extrapolated AIS records.

        Returns:
            (matched, unmatched_sar, unmatched_ais)
        """
        # 1. Extrapolate all AIS records to the exact SAR observation epoch
        extrapolated_ais = [self.extrapolate_dead_reckoning(rec, target_ts) for rec in ais_records]

        n_sar = len(sar_detections)
        n_ais = len(extrapolated_ais)

        if n_sar == 0:
            return [], [], extrapolated_ais
        if n_ais == 0:
            return [], sar_detections, []

        # 2. Build pairwise distance matrix (km)
        dist_matrix = np.full((n_sar, n_ais), 1e6, dtype=float)
        for i, s in enumerate(sar_detections):
            for j, a in enumerate(extrapolated_ais):
                d = self.haversine_km(s["lat"], s["lon"], a["lat"], a["lon"])
                if d <= gate_dist_km:
                    dist_matrix[i, j] = d

        # 3. Optimal bipartite matching via Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched: List[Dict[str, Any]] = []
        matched_sar_indices = set()
        matched_ais_indices = set()

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= gate_dist_km:
                matched_sar_indices.add(r)
                matched_ais_indices.add(c)
                sar_item = sar_detections[r]
                ais_item = extrapolated_ais[c]
                matched.append({
                    "sar_id": sar_item.get("detection_id", r + 1),
                    "mmsi": ais_item.get("mmsi", 0),
                    "sar_lat": sar_item["lat"],
                    "sar_lon": sar_item["lon"],
                    "ais_lat": ais_item["lat"],
                    "ais_lon": ais_item["lon"],
                    "sar_length_m": sar_item.get("length_m", 0.0),
                    "distance_km": float(dist_matrix[r, c]),
                })

        # 4. Identify unmatched detections
        unmatched_sar = [s for idx, s in enumerate(sar_detections) if idx not in matched_sar_indices]
        unmatched_ais = [a for idx, a in enumerate(extrapolated_ais) if idx not in matched_ais_indices]

        return matched, unmatched_sar, unmatched_ais
