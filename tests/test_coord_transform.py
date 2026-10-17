"""Kiểm thử đơn vị cho mô-đun chuyển đổi hệ tọa độ đa cảm biến."""

from src.coord_transform import CoordinateTransformer


def test_transformer_initialization_with_defaults_succeeds() -> None:
    # Arrange & Act
    transformer = CoordinateTransformer()

    # Assert
    assert transformer is not None
    assert transformer.platform_lat == 30.563549


def test_geodetic_to_radar_with_gt_coords_returns_expected_pixels() -> None:
    # Arrange
    transformer = CoordinateTransformer()
    lat, lon = 30.570142, 114.297007

    # Act
    xr, yr = transformer.geodetic_to_radar(lat, lon)

    # Assert
    assert 1600.0 <= xr <= 1850.0
    assert 1050.0 <= yr <= 1250.0


def test_radar_bbox_to_visible_gate_generates_valid_camera_gate() -> None:
    # Arrange
    transformer = CoordinateTransformer()
    bbox_rad = [1723.0, 1134.0, 29.0, 56.0]

    # Act
    gate_vis = transformer.radar_bbox_to_visible_gate(bbox_rad)

    # Assert
    assert len(gate_vis) == 4
    assert 600.0 <= gate_vis[0] <= 900.0
    assert 650.0 <= gate_vis[1] <= 850.0
    assert gate_vis[2] > 50.0
    assert gate_vis[3] > 20.0
