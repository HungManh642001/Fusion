import pytest
import numpy as np
from naval_isr.simulation.coastal_scenario import CoastalScenarioSimulator, TacticalVessel

def test_coastal_scenario_generation():
    sim = CoastalScenarioSimulator()
    vessels, sensor_data = sim.generate_scenario(duration_sec=100, dt=2.0)
    assert len(vessels) == 3
    # Vessel 1: Cooperative, Vessel 2: Spoofed, Vessel 3: Dark Warship
    types = [v.vessel_type for v in vessels]
    assert "cooperative" in types
    assert "ais_spoofed" in types
    assert "dark_warship" in types
    
    assert len(sensor_data["radar_measurements"]) > 0
    assert len(sensor_data["esm_measurements"]) > 0
    assert len(sensor_data["ais_messages"]) > 0

def test_vessel_sensor_properties():
    sim = CoastalScenarioSimulator()
    vessels, sensor_data = sim.generate_scenario(duration_sec=60, dt=2.0)
    
    # Check dark warship has no AIS but active emitter
    dark_warship = next(v for v in vessels if v.vessel_type == "dark_warship")
    assert not dark_warship.has_ais
    assert dark_warship.emitter_active
    
    # Check cooperative vessel has AIS and no spoofing
    coop = next(v for v in vessels if v.vessel_type == "cooperative")
    assert coop.has_ais
    assert not coop.ais_spoofed
    
    # Check spoofed vessel has AIS and ais_spoofed is True
    spoofed = next(v for v in vessels if v.vessel_type == "ais_spoofed")
    assert spoofed.has_ais
    assert spoofed.ais_spoofed
