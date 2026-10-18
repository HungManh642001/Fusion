import pytest
import numpy as np
from naval_isr.ai_inference.dsmt_fusion import DSmTEngine
from naval_isr.ai_inference.threat_assessor import ThreatAssessor

def test_pcr5_zero_paradox_high_conflict():
    engine = DSmTEngine(classes=["Destroyer", "Frigate", "Commercial"])
    # Sensor 1 (ESM): Strongly believes Destroyer (0.9), slight Commercial (0.1)
    m1 = {"Destroyer": 0.9, "Commercial": 0.1}
    # Sensor 2 (SAR): Strongly believes Frigate (0.9), slight Commercial (0.1)
    m2 = {"Frigate": 0.9, "Commercial": 0.1}

    # Under classical Dempster's rule, Commercial would paradoxically jump to 1.0.
    # Under DSmT PCR5, conflict is redistributed proportionally to Destroyer and Frigate!
    fused = engine.combine_pcr5(m1, m2)
    assert fused["Destroyer"] > 0.35
    assert fused["Frigate"] > 0.35
    assert fused["Commercial"] < 0.20  # Never experiences Zadeh's paradox!
    assert abs(sum(fused.values()) - 1.0) < 1e-5

def test_threat_assessor_scoring():
    assessor = ThreatAssessor(coastal_asset_loc=np.array([0.0, 0.0]))
    # Dark warship heading towards defense asset at high speed
    target_state = {
        "x": 5000.0, "y": 5000.0, "vx": -15.0, "vy": -15.0,  # moving inward towards [0, 0]
        "is_dark": True,
    }
    belief = {"Warship": 0.85, "Commercial": 0.15}
    assessment = assessor.compute_threat_priority_index(target_state, belief)
    
    assert assessment["tpi_score"] >= 75.0
    assert assessment["threat_level"] == "CRITICAL"
    assert "INTERCEPT" in assessment["countermeasure_action"] or "ENGAGE" in assessment["countermeasure_action"]

def test_threat_assessor_commercial_vessel():
    assessor = ThreatAssessor(coastal_asset_loc=np.array([0.0, 0.0]))
    # Slow cooperative merchant moving away
    target_state = {
        "x": 25000.0, "y": 25000.0, "vx": 10.0, "vy": 5.0,  # moving outward
        "is_dark": False,
    }
    belief = {"Warship": 0.05, "Commercial": 0.95}
    assessment = assessor.compute_threat_priority_index(target_state, belief)
    
    assert assessment["tpi_score"] < 40.0
    assert assessment["threat_level"] == "ROUTINE"
