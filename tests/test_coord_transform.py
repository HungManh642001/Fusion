import pytest
import numpy as np
from src.coord_transform import CoordinateTransformer

def test_coordinate_transformer_initialization():
    transformer = CoordinateTransformer()
    assert transformer is not None

def test_geodetic_to_radar_mapping():
    transformer = CoordinateTransformer()
    # MMSI 413806072 at 08:56:47
    lat, lon = 30.570142, 114.297007
    xr, yr = transformer.geodetic_to_radar(lat, lon)
    # Expected radar pixel coordinates around [1737.5, 1162.0] (+- 50px tolerance)
    assert 1600 <= xr <= 1850
    assert 1050 <= yr <= 1250

def test_radar_to_visible_gate():
    transformer = CoordinateTransformer()
    # Radar bbox around vessel [1723, 1134, 29, 56]
    bbox_rad = [1723, 1134, 29, 56]
    gate_vis = transformer.radar_bbox_to_visible_gate(bbox_rad)
    assert len(gate_vis) == 4
    # Expected visible bbox in 2560x1440 around [727, 761, 377, 61]
    assert 600 <= gate_vis[0] <= 900
    assert 650 <= gate_vis[1] <= 850
    assert gate_vis[2] > 50  # width
    assert gate_vis[3] > 20  # height
