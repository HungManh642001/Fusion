import pytest
import os
from naval_isr.sar_recon.dark_vessel_detector import DarkVesselDetector
from naval_isr.visualization.gis_map_renderer import GISTacticalMapRenderer

def test_dark_vessel_flagging():
    detector = DarkVesselDetector(min_warship_length_m=50.0)
    unmatched_sar = [
        {"detection_id": 101, "lat": 15.5, "lon": 112.5, "length_m": 120.0},  # Warship length -> Dark
        {"detection_id": 102, "lat": 15.6, "lon": 112.6, "length_m": 15.0},   # Small skiff
    ]
    unmatched_ais = [
        {"mmsi": 999111222, "lat": 16.0, "lon": 113.0, "sog_knots": 15.0}  # Ghost target
    ]
    matched = [
        {"sar_id": 1, "mmsi": 413806072, "sar_lat": 15.1, "sar_lon": 112.1, "distance_km": 0.2}
    ]
    results = detector.classify_vessels(matched=matched, unmatched_sar=unmatched_sar, unmatched_ais=unmatched_ais)
    assert len(results["dark_vessels"]) == 1
    assert results["dark_vessels"][0]["detection_id"] == 101
    assert len(results["spoofed_ghosts"]) == 1
    assert len(results["cooperative"]) == 1
    assert len(results["unidentified_small_craft"]) == 1

def test_gis_map_renderer(tmp_path):
    renderer = GISTacticalMapRenderer()
    classified_results = {
        "cooperative": [{"mmsi": 413806072, "sar_lat": 15.1, "sar_lon": 112.1, "sar_length_m": 150.0}],
        "dark_vessels": [{"detection_id": 101, "lat": 15.5, "lon": 112.5, "length_m": 120.0, "alert_level": "RED"}],
        "spoofed_ghosts": [{"mmsi": 999111222, "lat": 16.0, "lon": 113.0, "sog_knots": 15.0, "alert_level": "YELLOW"}],
        "unidentified_small_craft": [],
    }
    out_file = str(tmp_path / "test_map.html")
    rendered_path = renderer.render_interactive_map(classified_results, out_file)
    assert os.path.exists(rendered_path)
    with open(rendered_path, "r", encoding="utf-8") as f:
        html = f.read()
    assert "Dark Vessel" in html or "RED" in html
    assert "Ghost" in html or "YELLOW" in html
