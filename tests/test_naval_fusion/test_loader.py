"""Kiểm thử bộ nạp và tiền xử lý dữ liệu Autoferry (loader.py).

Module này kiểm thử khả năng đọc dữ liệu kịch bản JSON từ bộ chuẩn Autoferry,
chuyển đổi điểm đo dị thể (Radar, LiDAR, IR, EO) sang cấu trúc Detection chuẩn hóa,
bù vị trí tàu mẹ trong hệ NED, chuẩn hóa góc phương vị và tải Ground Truth.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.naval_fusion.loader import AutoferryLoader
from src.naval_fusion.models import Detection, SensorType


def test_load_scenario2_valid_returns_detections_and_ground_truth() -> None:
    """Kiểm tra nạp kịch bản scenario2 thực tế từ thư mục dataset."""
    # Arrange
    scenario_dir = Path("datasets/Autoferry-sensor_fusion_dataset/scenario2")
    if not scenario_dir.exists():
        pytest.skip("Autoferry scenario2 không tồn tại trên hệ thống kiểm thử.")

    loader = AutoferryLoader()

    # Act
    detections, ground_truth = loader.load_scenario(scenario_dir)

    # Assert
    assert len(detections) > 1000
    assert len(ground_truth) > 1000

    # Kiểm tra tính đơn điệu thời gian tăng dần của danh sách detections
    timestamps = [d.timestamp for d in detections]
    assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))

    # Kiểm tra sự hiện diện của đầy đủ 4 loại cảm biến dị thể
    sensor_types_present = {d.sensor_type for d in detections}
    assert SensorType.RADAR in sensor_types_present
    assert SensorType.LIDAR in sensor_types_present
    assert SensorType.IR in sensor_types_present
    assert SensorType.EO in sensor_types_present

    # Kiểm tra cấu trúc điểm đo Radar (2D Descartes NED)
    radar_dets = [d for d in detections if d.sensor_type == SensorType.RADAR]
    assert len(radar_dets) > 0
    assert radar_dets[0].measurement_ned.shape == (2,)
    assert radar_dets[0].measurement_ned.dtype == np.float64
    assert len(radar_dets[0].raw_measurement) == 2

    # Kiểm tra cấu trúc điểm đo LiDAR (2D Descartes NED)
    lidar_dets = [d for d in detections if d.sensor_type == SensorType.LIDAR]
    assert len(lidar_dets) > 0
    assert lidar_dets[0].measurement_ned.shape == (2,)
    assert lidar_dets[0].measurement_ned.dtype == np.float64
    assert len(lidar_dets[0].raw_measurement) == 2

    # Kiểm tra cấu trúc điểm đo Hồng ngoại IR (1D Bearing NED [-pi, pi])
    ir_dets = [d for d in detections if d.sensor_type == SensorType.IR]
    assert len(ir_dets) > 0
    assert ir_dets[0].measurement_ned.shape == (1,)
    assert -np.pi <= ir_dets[0].measurement_ned[0] <= np.pi
    assert len(ir_dets[0].raw_measurement) == 1

    # Kiểm tra cấu trúc điểm đo Quang học EO (1D Bearing NED [-pi, pi])
    eo_dets = [d for d in detections if d.sensor_type == SensorType.EO]
    assert len(eo_dets) > 0
    assert eo_dets[0].measurement_ned.shape == (1,)
    assert -np.pi <= eo_dets[0].measurement_ned[0] <= np.pi
    assert len(eo_dets[0].raw_measurement) == 1


def test_load_scenario_nonexistent_directory_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Kiểm tra ngoại lệ FileNotFoundError khi đường dẫn kịch bản không tồn tại."""
    # Arrange
    loader = AutoferryLoader()
    non_existent = tmp_path / "scenario_missing"

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Thư mục kịch bản không tồn tại"):
        loader.load_scenario(non_existent)


def test_load_scenario_missing_detections_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Kiểm tra ngoại lệ FileNotFoundError khi thiếu file *_detections.json."""
    # Arrange
    loader = AutoferryLoader()
    scenario_dir = tmp_path / "scenario_no_det"
    scenario_dir.mkdir()
    (scenario_dir / "scenario_no_det_groundTruth.json").write_text(
        "[]", encoding="utf-8"
    )

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tệp detections JSON"):
        loader.load_scenario(scenario_dir)


def test_load_scenario_missing_ground_truth_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Kiểm tra ngoại lệ FileNotFoundError khi thiếu file *_groundTruth.json."""
    # Arrange
    loader = AutoferryLoader()
    scenario_dir = tmp_path / "scenario_no_gt"
    scenario_dir.mkdir()
    (scenario_dir / "scenario_no_gt_detections.json").write_text("[]", encoding="utf-8")

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tệp groundTruth JSON"):
        loader.load_scenario(scenario_dir)


def test_parse_cartesian_single_point_compensates_correctly() -> None:
    """Kiểm tra phân tích điểm đo Radar dạng 1 điểm [x, y]."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 2,
        "time": 1620114890.0,
        "ownshipPosition": [100.0, -200.0],
        "measurement": [50.0, 30.0],
    }

    # Act
    dets = loader.parse_single_record(record)

    # Assert
    assert len(dets) == 1
    det = dets[0]
    assert det.sensor_type == SensorType.RADAR
    assert det.timestamp == 1620114890.0
    assert det.ownship_pos == (100.0, -200.0)
    assert det.raw_measurement == [50.0, 30.0]
    np.testing.assert_allclose(det.measurement_ned, np.array([150.0, -170.0]))


def test_parse_cartesian_matlab_matrix_2xk_expands_all_points() -> None:
    """Kiểm tra phân tích điểm đo 2xK MATLAB: row 0 là X, row 1 là Y."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 1,
        "time": 1620114895.0,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [
            [10.0, 20.0, 30.0],  # xs
            [100.0, 200.0, 300.0],  # ys
        ],
    }

    # Act
    dets = loader.parse_single_record(record)

    # Assert
    assert len(dets) == 3
    assert dets[0].raw_measurement == [10.0, 100.0]
    np.testing.assert_allclose(dets[0].measurement_ned, np.array([10.0, 100.0]))
    assert dets[1].raw_measurement == [20.0, 200.0]
    np.testing.assert_allclose(dets[1].measurement_ned, np.array([20.0, 200.0]))
    assert dets[2].raw_measurement == [30.0, 300.0]
    np.testing.assert_allclose(dets[2].measurement_ned, np.array([30.0, 300.0]))


def test_parse_cartesian_point_list_nx2_expands_all_points() -> None:
    """Kiểm tra phân tích điểm đo dạng danh sách nhiều điểm [[x1, y1], ...]."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 2,
        "time": 1620114896.0,
        "ownshipPosition": [10.0, 20.0],
        "measurement": [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
    }

    # Act
    dets = loader.parse_single_record(record)

    # Assert
    assert len(dets) == 3
    assert dets[0].raw_measurement == [1.0, 2.0]
    np.testing.assert_allclose(dets[0].measurement_ned, np.array([11.0, 22.0]))
    assert dets[1].raw_measurement == [3.0, 4.0]
    np.testing.assert_allclose(dets[1].measurement_ned, np.array([13.0, 24.0]))
    assert dets[2].raw_measurement == [5.0, 6.0]
    np.testing.assert_allclose(dets[2].measurement_ned, np.array([15.0, 26.0]))


def test_parse_cartesian_empty_returns_empty_list() -> None:
    """Kiểm tra khi điểm đo Radar/LiDAR rỗng ([]) thì không tạo Detection."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 2,
        "time": 1620114897.0,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [],
    }

    # Act
    dets = loader.parse_single_record(record)

    # Assert
    assert dets == []


def test_parse_bearing_single_scalar_normalizes_angle() -> None:
    """Kiểm tra phân tích điểm đo góc phương vị IR dạng số thực đơn."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 3,
        "time": 1620114898.0,
        "ownshipPosition": [50.0, 50.0],
        "measurement": np.pi + 0.2,  # Vượt quá pi, cần chuẩn hóa
    }

    # Act
    dets = loader.parse_single_record(record)

    # Assert
    assert len(dets) == 1
    det = dets[0]
    assert det.sensor_type == SensorType.IR
    assert det.timestamp == 1620114898.0
    assert det.ownship_pos == (50.0, 50.0)
    assert np.isclose(det.raw_measurement[0], np.pi + 0.2)
    assert np.isclose(det.measurement_ned[0], -np.pi + 0.2)


def test_parse_bearing_list_of_angles_expands_all() -> None:
    """Kiểm tra phân tích điểm đo EO với danh sách nhiều góc phương vị."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 4,
        "time": 1620114899.0,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [0.5, -1.2, 2.8],
    }

    # Act
    dets = loader.parse_single_record(record)

    # Assert
    assert len(dets) == 3
    for i, expected_raw in enumerate([0.5, -1.2, 2.8]):
        assert dets[i].sensor_type == SensorType.EO
        assert np.isclose(dets[i].raw_measurement[0], expected_raw)
        assert np.isclose(dets[i].measurement_ned[0], expected_raw)


def test_parse_bearing_empty_returns_empty_list() -> None:
    """Kiểm tra khi điểm đo IR/EO rỗng thì trả về danh sách rỗng."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 3,
        "time": 1620114900.0,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [],
    }

    # Act
    dets = loader.parse_single_record(record)

    # Assert
    assert dets == []


def test_parse_record_unknown_sensor_id_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi gặp sensorID không hợp lệ."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 99,
        "time": 1620114901.0,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [10.0, 20.0],
    }

    # Act & Assert
    with pytest.raises(ValueError, match="Không nhận diện được sensorID"):
        loader.parse_single_record(record)


def test_parse_record_invalid_ownship_position_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi ownshipPosition không có đúng 2 phần tử."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 1,
        "time": 1620114902.0,
        "ownshipPosition": [0.0],
        "measurement": [10.0, 20.0],
    }

    # Act & Assert
    with pytest.raises(ValueError, match="ownshipPosition phải có đúng 2 phần tử"):
        loader.parse_single_record(record)


def test_parse_record_invalid_cartesian_measurement_shape_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi điểm đo Descartes có kích thước sai."""
    # Arrange
    loader = AutoferryLoader()
    record = {
        "sensorID": 1,
        "time": 1620114903.0,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [10.0],  # Chỉ có 1 phần tử cho sensor 2D
    }

    # Act & Assert
    with pytest.raises(ValueError, match="Kích thước điểm đo Descartes không hợp lệ"):
        loader.parse_single_record(record)


def test_load_scenario_from_string_path(tmp_path: Path) -> None:
    """Kiểm tra nhận đường dẫn dạng chuỗi (str) hợp lệ."""
    # Arrange
    loader = AutoferryLoader()
    scen_dir = tmp_path / "scenario_str"
    scen_dir.mkdir()
    dets_content = [
        {
            "sensorID": 2,
            "time": 10.0,
            "ownshipPosition": [0.0, 0.0],
            "measurement": [5.0, 5.0],
        }
    ]
    gt_content = [{"targetID": 1, "position": [5.0, 5.0, 0], "time": 10.0}]
    (scen_dir / "scenario_str_detections.json").write_text(
        json.dumps(dets_content), encoding="utf-8"
    )
    (scen_dir / "scenario_str_groundTruth.json").write_text(
        json.dumps(gt_content), encoding="utf-8"
    )

    # Act
    dets, gt = loader.load_scenario(str(scen_dir))

    # Assert
    assert len(dets) == 1
    assert len(gt) == 1
    assert isinstance(dets[0], Detection)
