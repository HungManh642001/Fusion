"""Dark Vessel and AIS Deception (Ghost Ship) Detection Engine.

Classifies maritime targets by fusing Spaceborne SAR physical detections with AIS transponder messages:
1. Cooperative Vessels: Confirmed physical target matching valid AIS broadcast.
2. Dark Vessels (RED ALERT): Large metallic vessel detected on SAR with AIS deliberately disabled.
3. AIS Spoofed Ghosts (YELLOW ALERT): AIS broadcast active where no physical vessel is observed by SAR.
4. Unidentified Small Craft: Small vessels (<50m, wooden skiffs) lacking mandatory Class A AIS.
"""

from typing import List, Dict, Any, Optional


class DarkVesselDetector:
    """Classifies maritime contacts into tactical categories and assigns military alert levels."""

    def __init__(self, min_warship_length_m: float = 50.0):
        self.min_warship_length_m = min_warship_length_m

    def classify_vessels(
        self,
        matched: List[Dict[str, Any]],
        unmatched_sar: List[Dict[str, Any]],
        unmatched_ais: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Classify targets into cooperative, dark vessels, ghost ships, and small craft."""
        cooperative_list = []
        for m in matched:
            item = dict(m)
            item["classification"] = "cooperative"
            item["alert_level"] = "GREEN"
            cooperative_list.append(item)

        dark_vessels = []
        small_craft = []
        for s in unmatched_sar:
            length = float(s.get("length_m", 0.0))
            item = dict(s)
            if length >= self.min_warship_length_m:
                item["classification"] = "dark_vessel"
                item["alert_level"] = "RED"
                item["threat_type"] = "Unannounced Warship / Submarine Surface / Smuggling Mother Ship"
                dark_vessels.append(item)
            else:
                item["classification"] = "small_craft"
                item["alert_level"] = "INFO"
                item["threat_type"] = "Non-SOLAS Small Craft / Fishing Skiff"
                small_craft.append(item)

        spoofed_ghosts = []
        for a in unmatched_ais:
            item = dict(a)
            item["classification"] = "spoofed_ghost"
            item["alert_level"] = "YELLOW"
            item["threat_type"] = "AIS Deception / Electronic Ghost Target"
            spoofed_ghosts.append(item)

        return {
            "cooperative": cooperative_list,
            "dark_vessels": dark_vessels,
            "spoofed_ghosts": spoofed_ghosts,
            "unidentified_small_craft": small_craft,
        }
