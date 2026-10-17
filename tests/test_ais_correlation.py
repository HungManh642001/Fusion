import pytest
import numpy as np
from src.naval_isr.sar_recon.ais_correlation import AISCorrelationEngine
from src.naval_isr.sar_recon.sar_preprocessor import SARPreprocessor

def test_ais_extrapolation_matching():
    engine = AISCorrelationEngine()
    ais_records = [{
        "mmsi": 413806072, "lat": 30.5600, "lon": 114.2900, "sog": 10.0, "cog": 0.0, "ts": 1000
    }]
    # 100 seconds later moving due North (cog 0, sog 10 knots = ~5.14 m/s -> ~514m north ~ 0.00463 deg)
    sar_detections = [{
        "detection_id": 1, "lat": 30.5646, "lon": 114.2900, "length_m": 85.0
    }]
    matched, dark, ghosts = engine.match_sar_to_ais(sar_detections, ais_records, target_ts=1100, gate_dist_km=1.5)
    assert len(matched) == 1
    assert matched[0]["sar_id"] == 1
    assert matched[0]["mmsi"] == 413806072
    assert len(dark) == 0
    assert len(ghosts) == 0

def test_sar_preprocessor_calibration_and_cfar():
    prep = SARPreprocessor()
    # Simulated 100x100 SAR chip with ocean background and 1 bright target
    raw_dn = np.ones((100, 100)) * 50.0  # background ocean
    raw_dn[50, 50] = 500.0  # bright metallic target
    
    sigma0 = prep.calibrate_sigma0(raw_dn, ks=30.0)
    assert sigma0.shape == (100, 100)
    assert sigma0[50, 50] > sigma0[10, 10]
    
    # Run CFAR
    targets = prep.extract_cfar_targets(sigma0, guard_size=3, train_size=7, threshold_factor=2.5)
    assert len(targets) >= 1
    t = targets[0]
    assert abs(t["pixel_r"] - 50) <= 1
    assert abs(t["pixel_c"] - 50) <= 1
