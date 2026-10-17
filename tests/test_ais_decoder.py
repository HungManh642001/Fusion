import pytest
import os
from src.ais_decoder import AISDecoder, AISMessage, extrapolate_kinematic

def test_ais_decoder_reads_file():
    ais_path = "datasets/scene01/AIS.txt"
    assert os.path.exists(ais_path)
    decoder = AISDecoder()
    messages = decoder.decode_file(ais_path)
    assert len(messages) > 1000
    
    # Check that MMSI 413806072 is present
    target_msgs = [m for m in messages if m.mmsi == 413806072]
    assert len(target_msgs) > 0
    first = target_msgs[0]
    assert 30.55 < first.lat < 30.59
    assert 114.28 < first.lon < 114.32
    assert first.sog > 4.0

def test_extrapolate_kinematic():
    # Test kinematic dead reckoning (Eq. 8 & 9)
    # Moving due North (COG = 0), SOG = 10 knots = 5.144 m/s, dt = 100s
    # Distance = 514.4 m -> delta lat = 514.4 / (pi * R / 180) ~ 0.004626 deg
    msg = AISMessage(
        ts=1000000,
        mmsi=123456789,
        lat=30.0,
        lon=114.0,
        sog=10.0,
        cog=0.0
    )
    new_lat, new_lon = extrapolate_kinematic(msg, 1000000 + 100000) # +100s
    assert new_lat > 30.0
    assert pytest.approx(new_lon, rel=1e-5) == 114.0
    # ~514.4 meters / 111139 m/deg ~ 0.004628 deg
    assert pytest.approx(new_lat - 30.0, abs=1e-3) == 0.00463
