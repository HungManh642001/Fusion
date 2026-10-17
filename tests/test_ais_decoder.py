"""Kiểm thử đơn vị cho mô-đun giải mã AIS và ngoại suy động học."""

import os

import pytest

from src.ais_decoder import AISDecoder, AISMessage, extrapolate_kinematic


def test_decode_file_with_valid_log_returns_messages() -> None:
    # Arrange
    ais_path = "datasets/scene01/AIS.txt"
    assert os.path.exists(ais_path)
    decoder = AISDecoder()

    # Act
    messages = decoder.decode_file(ais_path)

    # Assert
    assert len(messages) > 1000
    target_msgs = [m for m in messages if m.mmsi == 413806072]
    assert len(target_msgs) > 0
    first = target_msgs[0]
    assert 30.55 < first.lat < 30.59
    assert 114.28 < first.lon < 114.32
    assert first.sog > 4.0


def test_extrapolate_kinematic_with_linear_motion_returns_coordinates() -> None:
    # Arrange
    msg = AISMessage(
        ts=1000000,
        mmsi=123456789,
        lat=30.0,
        lon=114.0,
        sog=10.0,
        cog=0.0,
    )
    target_ts = 1000000 + 100000

    # Act
    new_lat, new_lon = extrapolate_kinematic(msg, target_ts=target_ts)

    # Assert
    assert new_lat > 30.0
    assert pytest.approx(new_lon, rel=1e-5) == 114.0
    assert pytest.approx(new_lat - 30.0, abs=1e-3) == 0.00463
