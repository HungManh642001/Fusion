"""Kiểm thử tích hợp phân hệ hợp nhất động học hải quân (NavalFusionPipeline).

Kiểm tra luồng xử lý end-to-end trên tập dữ liệu chuẩn Autoferry Benchmark Dataset,
đối chứng độ chính xác vị trí RMSE, tính toàn vẹn quỹ đạo, và giao diện dòng lệnh CLI.
"""

import os
from pathlib import Path

import pytest

from src.naval_fusion.models import EvaluationMetrics, SensorType, TrackState
from src.naval_fusion.pipeline import NavalFusionPipeline


def test_pipeline_run_scenario2() -> None:
    """Kiểm tra pipeline chạy tích hợp toàn diện trên Autoferry scenario2."""
    scenario_dir = "datasets/Autoferry-sensor_fusion_dataset/scenario2"
    if not os.path.exists(scenario_dir):
        pytest.skip("Thư mục Autoferry scenario2 không tồn tại")

    pipeline = NavalFusionPipeline()
    history, metrics = pipeline.run(scenario_dir)

    # 1. Kiểm tra cấu trúc dữ liệu trả về
    assert isinstance(history, dict)
    assert len(history) > 0
    assert isinstance(metrics, EvaluationMetrics)

    # 2. Hệ thống phải bám bắt thành công ít nhất 2 tàu mục tiêu
    assert metrics.num_confirmed_tracks >= 2

    # 3. Sai số RMSE vị trí phải hội tụ dưới ngưỡng cho phép (< 10 mét trên trường 500m)
    assert metrics.rmse_position < 10.0
    assert metrics.rmse_velocity < 10.0

    # 4. Tỷ lệ duy trì bám vết (Mostly Tracked) phải đạt trên 50%
    assert metrics.mostly_tracked_ratio >= 0.5

    # 5. Kiểm tra các vết được ghi nhận trong lịch sử đều là vết đã xác nhận
    for t_step, tracks in history.items():
        assert isinstance(t_step, float)
        assert len(tracks) > 0
        for trk in tracks:
            assert isinstance(trk, TrackState)
            assert trk.is_confirmed is True
            assert trk.state_vector.shape == (4,)
            assert trk.covariance.shape == (4, 4)


def test_pipeline_enabled_sensors_radar_only() -> None:
    """Kiểm tra chạy pipeline với chế độ chỉ dùng cảm biến Radar (Radar-only)."""
    scenario_dir = "datasets/Autoferry-sensor_fusion_dataset/scenario2"
    if not os.path.exists(scenario_dir):
        pytest.skip("Thư mục Autoferry scenario2 không tồn tại")

    pipeline = NavalFusionPipeline()
    history, metrics = pipeline.run(
        scenario_dir,
        enabled_sensors={SensorType.RADAR},
    )

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.num_confirmed_tracks >= 2
    assert metrics.rmse_position < 15.0

    # Tất cả các vết xác nhận chỉ được đóng góp bởi RADAR
    for tracks in history.values():
        for trk in tracks:
            assert trk.contributing_sensors == {SensorType.RADAR}


def test_pipeline_invalid_scenario_dir_raises_file_not_found() -> None:
    """Kiểm tra đường dẫn không tồn tại sẽ báo lỗi FileNotFoundError."""
    pipeline = NavalFusionPipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.run("datasets/nonexistent_scenario_xyz")


def test_pipeline_empty_detections_returns_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kiểm tra xử lý biên khi tập điểm đo rỗng."""

    scenario_dir = "datasets/Autoferry-sensor_fusion_dataset/scenario2"
    if not os.path.exists(scenario_dir):
        pytest.skip("Thư mục Autoferry scenario2 không tồn tại")

    pipeline = NavalFusionPipeline()

    # Giả lập loader trả về danh sách điểm đo rỗng
    monkeypatch.setattr(pipeline.loader, "load_scenario", lambda _path: ([], []))

    history, metrics = pipeline.run(scenario_dir)
    assert history == {}
    assert metrics.num_confirmed_tracks == 0
    assert metrics.rmse_position == 0.0


def test_pipeline_ablation_run_comparison() -> None:
    """Kiểm tra chạy so sánh đối chứng ablation giữa Radar-only và All sensors."""
    scenario_dir = "datasets/Autoferry-sensor_fusion_dataset/scenario2"
    if not os.path.exists(scenario_dir):
        pytest.skip("Thư mục Autoferry scenario2 không tồn tại")

    pipeline = NavalFusionPipeline()

    _, metrics_radar = pipeline.run(
        scenario_dir, enabled_sensors={SensorType.RADAR}
    )
    _, metrics_all = pipeline.run(scenario_dir, enabled_sensors=None)

    # Chế độ All sensors phải tận dụng được dữ liệu quang nhiệt và LiDAR
    assert metrics_all.num_confirmed_tracks >= 2
    assert metrics_radar.num_confirmed_tracks >= 2
    assert metrics_all.rmse_position < 10.0


def test_cli_runner_scenario2_eval(tmp_path: Path) -> None:
    """Kiểm tra thực thi CLI với cờ --scenario và --eval."""
    import run_naval_fusion

    output_file = tmp_path / "report_eval.md"
    exit_code = run_naval_fusion.main(
        [
            "--scenario",
            "scenario2",
            "--eval",
            "--output",
            str(output_file),
        ]
    )

    assert exit_code == 0
    assert output_file.is_file()
    content = output_file.read_text(encoding="utf-8")
    assert "Báo Cáo Đánh Giá Hiệu Năng" in content
    assert "scenario2" in content
    assert "RMSE Vị trí" in content


def test_cli_runner_scenario2_ablation(tmp_path: Path) -> None:
    """Kiểm tra thực thi CLI với cờ --scenario và --ablation."""
    import run_naval_fusion

    output_file = tmp_path / "report_ablation.md"
    exit_code = run_naval_fusion.main(
        [
            "--scenario",
            "scenario2",
            "--ablation",
            "--output",
            str(output_file),
        ]
    )

    assert exit_code == 0
    assert output_file.is_file()
    content = output_file.read_text(encoding="utf-8")
    assert "Radar-Only" in content
    assert "Radar + IR" in content
    assert "Hợp nhất Toàn diện" in content
