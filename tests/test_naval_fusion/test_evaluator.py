"""Kiểm thử bộ đánh giá định lượng phân hệ hợp nhất hải quân (evaluator.py).

Module này kiểm thử tính toán sai số vị trí RMSE, vận tốc RMSE, số vết xác nhận,
số lần nhảy định danh vết (ID Switches), tỷ lệ bám vết liên tục (Mostly Tracked - MT),
và tạo báo cáo đánh giá định dạng Markdown.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.naval_fusion.evaluator import NavalEvaluator
from src.naval_fusion.loader import AutoferryLoader
from src.naval_fusion.models import EvaluationMetrics, SensorType, TrackState


def test_calc_rmse_pos_valid_data_returns_correct_rmse() -> None:
    """Kiểm tra tính RMSE vị trí trên dữ liệu 2D chuẩn."""
    # Arrange
    evaluator = NavalEvaluator()
    est_pos = np.array([[10.0, 20.0], [15.0, 25.0]], dtype=np.float64)
    gt_pos = np.array([[10.0, 23.0], [15.0, 21.0]], dtype=np.float64)
    # Sai số: điểm 1 lệch 3m (3^2=9), điểm 2 lệch 4m (4^2=16).
    # RMSE = sqrt((9 + 16)/2) = sqrt(12.5) = 3.5355339 m

    # Act
    rmse = evaluator.calc_rmse_pos(est_pos, gt_pos)

    # Assert
    assert np.isclose(rmse, 3.5355339)


def test_calc_rmse_pos_empty_arrays_returns_zero() -> None:
    """Kiểm tra tính RMSE vị trí khi mảng đầu vào rỗng."""
    # Arrange
    evaluator = NavalEvaluator()
    est_pos = np.empty((0, 2), dtype=np.float64)
    gt_pos = np.empty((0, 2), dtype=np.float64)

    # Act
    rmse = evaluator.calc_rmse_pos(est_pos, gt_pos)

    # Assert
    assert rmse == 0.0


def test_calc_rmse_pos_single_point_returns_distance() -> None:
    """Kiểm tra tính RMSE vị trí với điểm đơn 1D vector."""
    # Arrange
    evaluator = NavalEvaluator()
    est_pos = np.array([10.0, 20.0], dtype=np.float64)
    gt_pos = np.array([10.0, 24.0], dtype=np.float64)

    # Act
    rmse = evaluator.calc_rmse_pos(est_pos, gt_pos)

    # Assert
    assert np.isclose(rmse, 4.0)


def test_calc_rmse_pos_shape_mismatch_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi kích thước mảng vị trí không tương thích."""
    # Arrange
    evaluator = NavalEvaluator()
    est_pos = np.array([[10.0, 20.0], [15.0, 25.0]], dtype=np.float64)
    gt_pos = np.array([[10.0, 20.0]], dtype=np.float64)

    # Act & Assert
    with pytest.raises(ValueError, match="bằng nhau"):
        evaluator.calc_rmse_pos(est_pos, gt_pos)


def test_calc_rmse_vel_valid_data_returns_correct_rmse() -> None:
    """Kiểm tra tính RMSE vận tốc trên dữ liệu 2D chuẩn."""
    # Arrange
    evaluator = NavalEvaluator()
    est_vel = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    gt_vel = np.array([[2.0, 1.0], [1.0, 5.0]], dtype=np.float64)
    # Điểm 1: (0^2 + 2^2) = 4; Điểm 2: (3^2 + 0^2) = 9
    # RMSE = sqrt((4 + 9)/2) = sqrt(6.5) = 2.5495097568

    # Act
    rmse = evaluator.calc_rmse_vel(est_vel, gt_vel)

    # Assert
    assert np.isclose(rmse, np.sqrt(6.5))


def test_calc_rmse_vel_empty_arrays_returns_zero() -> None:
    """Kiểm tra tính RMSE vận tốc khi mảng đầu vào rỗng."""
    # Arrange
    evaluator = NavalEvaluator()
    est_vel = np.empty((0, 2), dtype=np.float64)
    gt_vel = np.empty((0, 2), dtype=np.float64)

    # Act
    rmse = evaluator.calc_rmse_vel(est_vel, gt_vel)

    # Assert
    assert rmse == 0.0


def test_calc_rmse_vel_shape_mismatch_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi kích thước mảng vận tốc không tương thích."""
    # Arrange
    evaluator = NavalEvaluator()
    est_vel = np.array([[1.0, 2.0]], dtype=np.float64)
    gt_vel = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    # Act & Assert
    with pytest.raises(ValueError, match="bằng nhau"):
        evaluator.calc_rmse_vel(est_vel, gt_vel)


def test_evaluate_tracks_perfect_tracking_returns_zero_rmse_and_full_mt() -> None:
    """Kiểm tra đánh giá khi các vết bám chính xác 100% vào ground truth."""
    # Arrange
    evaluator = NavalEvaluator()
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Ground truth gồm 2 mục tiêu di chuyển thẳng
    ground_truth: list[list[dict[str, Any]]] = []
    confirmed_history: dict[float, list[TrackState]] = {}

    for t in timestamps:
        gt_frame = [
            {
                "targetID": 1,
                "position": [100.0 + 2.0 * t, 200.0, 0.0],
                "velocity": [2.0, 0.0],
                "time": t,
            },
            {
                "targetID": 2,
                "position": [-50.0, -100.0 + 1.0 * t, 0.0],
                "velocity": [0.0, 1.0],
                "time": t,
            },
        ]
        ground_truth.append(gt_frame)

        # Track 1 và Track 2 ước lượng hoàn hảo
        trk1 = TrackState(
            track_id=1,
            state_vector=np.array([100.0 + 2.0 * t, 200.0, 2.0, 0.0], dtype=np.float64),
            covariance=np.eye(4, dtype=np.float64),
            last_updated=t,
            hits=5,
            misses=0,
            is_confirmed=True,
            contributing_sensors={SensorType.RADAR},
        )
        trk2 = TrackState(
            track_id=2,
            state_vector=np.array(
                [-50.0, -100.0 + 1.0 * t, 0.0, 1.0], dtype=np.float64
            ),
            covariance=np.eye(4, dtype=np.float64),
            last_updated=t,
            hits=5,
            misses=0,
            is_confirmed=True,
            contributing_sensors={SensorType.RADAR},
        )
        confirmed_history[t] = [trk1, trk2]

    # Act
    metrics = evaluator.evaluate_tracks(confirmed_history, ground_truth)

    # Assert
    assert np.isclose(metrics.rmse_position, 0.0, atol=1e-5)
    assert np.isclose(metrics.rmse_velocity, 0.0, atol=1e-5)
    assert metrics.num_confirmed_tracks == 2
    assert metrics.id_switches == 0
    assert metrics.mostly_tracked_ratio == 1.0


def test_evaluate_tracks_id_switch_detected_increments_idsw() -> None:
    """Kiểm tra phát hiện hiện tượng nhảy định danh vết (ID Switch)."""
    # Arrange
    evaluator = NavalEvaluator()

    # Mục tiêu 1 tại t=1.0 được gán vết ID=10, nhưng tại t=2.0 bị tráo sang ID=20
    ground_truth = [
        [{"targetID": 1, "position": [10.0, 20.0, 0.0], "time": 1.0}],
        [{"targetID": 1, "position": [12.0, 20.0, 0.0], "time": 2.0}],
    ]
    track_t1 = TrackState(
        track_id=10,
        state_vector=np.array([10.0, 20.0, 2.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64),
        last_updated=1.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    track_t2 = TrackState(
        track_id=20,  # Nhảy ID từ 10 -> 20
        state_vector=np.array([12.0, 20.0, 2.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64),
        last_updated=2.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    confirmed_history = {1.0: [track_t1], 2.0: [track_t2]}

    # Act
    metrics = evaluator.evaluate_tracks(confirmed_history, ground_truth)

    # Assert
    assert metrics.id_switches == 1
    assert metrics.num_confirmed_tracks == 2


def test_evaluate_tracks_distance_gating_rejects_distant_tracks() -> None:
    """Kiểm tra loại bỏ các vết nằm ngoài ngưỡng khoảng cách distance_threshold."""
    # Arrange
    evaluator = NavalEvaluator()
    ground_truth = [
        [{"targetID": 1, "position": [10.0, 10.0, 0.0], "time": 1.0}],
    ]
    # Vết ở cách xa 100m (ngưỡng mặc định 25m)
    distant_track = TrackState(
        track_id=99,
        state_vector=np.array([110.0, 10.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64),
        last_updated=1.0,
        hits=4,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    confirmed_history = {1.0: [distant_track]}

    # Act
    metrics = evaluator.evaluate_tracks(
        confirmed_history, ground_truth, distance_threshold=25.0
    )

    # Assert
    # Vết không được ghép -> số điểm ghép = 0, RMSE = 0.0, MT = 0.0
    assert metrics.mostly_tracked_ratio == 0.0
    assert metrics.rmse_position == 0.0
    assert metrics.details["matched_points"] == 0


def test_evaluate_tracks_mostly_tracked_ratio_computed_correctly() -> None:
    """Kiểm tra tính tỷ lệ Mostly Tracked (>=80% thời gian bám vết)."""
    # Arrange
    evaluator = NavalEvaluator()
    # 10 mốc thời gian
    timestamps = [float(i) for i in range(10)]
    ground_truth: list[list[dict[str, Any]]] = []
    confirmed_history: dict[float, list[TrackState]] = {}

    # Mục tiêu 1 xuất hiện cả 10 bước, được bám 9 bước (90% >= 80% -> MT)
    # Mục tiêu 2 xuất hiện cả 10 bước, chỉ được bám 5 bước (50% < 80% -> Not MT)
    for t in timestamps:
        ground_truth.append(
            [
                {"targetID": 1, "position": [0.0, t * 10.0, 0.0], "time": t},
                {"targetID": 2, "position": [500.0, t * 10.0, 0.0], "time": t},
            ]
        )
        tracks: list[TrackState] = []
        if t < 9.0:
            tracks.append(
                TrackState(
                    track_id=1,
                    state_vector=np.array([0.0, t * 10.0, 0.0, 10.0], dtype=np.float64),
                    covariance=np.eye(4, dtype=np.float64),
                    last_updated=t,
                    hits=5,
                    misses=0,
                    is_confirmed=True,
                    contributing_sensors={SensorType.RADAR},
                )
            )
        if t < 5.0:
            tracks.append(
                TrackState(
                    track_id=2,
                    state_vector=np.array(
                        [500.0, t * 10.0, 0.0, 10.0], dtype=np.float64
                    ),
                    covariance=np.eye(4, dtype=np.float64),
                    last_updated=t,
                    hits=5,
                    misses=0,
                    is_confirmed=True,
                    contributing_sensors={SensorType.RADAR},
                )
            )
        confirmed_history[t] = tracks

    # Act
    metrics = evaluator.evaluate_tracks(confirmed_history, ground_truth)

    # Assert
    # 1/2 mục tiêu đạt chuẩn MT -> MT ratio = 0.5
    assert metrics.mostly_tracked_ratio == 0.5


def test_evaluate_tracks_empty_inputs_returns_zero_metrics() -> None:
    """Kiểm tra xử lý biên khi cả hai tập dữ liệu đầu vào đều rỗng."""
    # Arrange
    evaluator = NavalEvaluator()

    # Act
    metrics = evaluator.evaluate_tracks({}, [])

    # Assert
    assert metrics.rmse_position == 0.0
    assert metrics.rmse_velocity == 0.0
    assert metrics.num_confirmed_tracks == 0
    assert metrics.id_switches == 0
    assert metrics.mostly_tracked_ratio == 0.0


def test_evaluate_tracks_flat_ground_truth_list_parses_and_evaluates() -> None:
    """Kiểm tra khả năng tiếp nhận Ground Truth dạng danh sách phẳng các từ điển."""
    # Arrange
    evaluator = NavalEvaluator()
    flat_gt: list[dict[str, Any]] = [
        {"target_id": 1, "position": [10.0, 20.0], "time": 1.0, "velocity": [1.0, 0.0]},
        {"target_id": 1, "position": [11.0, 20.0], "time": 2.0, "velocity": [1.0, 0.0]},
    ]
    trk = TrackState(
        track_id=1,
        state_vector=np.array([10.0, 20.0, 1.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64),
        last_updated=1.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    confirmed_history = {1.0: [trk]}

    # Act
    metrics = evaluator.evaluate_tracks(confirmed_history, flat_gt)

    # Assert
    assert metrics.num_confirmed_tracks == 1
    assert np.isclose(metrics.rmse_position, 0.0)


def test_evaluate_scenario_alias_matches_evaluate_tracks() -> None:
    """Kiểm tra phương thức bí danh evaluate_scenario trả về kết quả tương đương."""
    # Arrange
    evaluator = NavalEvaluator()
    gt = [[{"targetID": 1, "position": [10.0, 20.0, 0.0], "time": 1.0}]]
    trk = TrackState(
        track_id=1,
        state_vector=np.array([10.0, 20.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64),
        last_updated=1.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    confirmed_history = {1.0: [trk]}

    # Act
    m1 = evaluator.evaluate_tracks(confirmed_history, gt)
    m2 = evaluator.evaluate_scenario(confirmed_history, gt)

    # Assert
    assert m1.rmse_position == m2.rmse_position
    assert m1.num_confirmed_tracks == m2.num_confirmed_tracks


def test_generate_markdown_report_valid_metrics_produces_table() -> None:
    """Kiểm tra xuất báo cáo so sánh bảng Markdown chuẩn định dạng."""
    # Arrange
    evaluator = NavalEvaluator()
    results = {
        "Radar-Only": EvaluationMetrics(
            rmse_position=4.25,
            rmse_velocity=0.85,
            num_confirmed_tracks=2,
            id_switches=1,
            mostly_tracked_ratio=0.85,
        ),
        "Radar + IR": EvaluationMetrics(
            rmse_position=2.80,
            rmse_velocity=0.62,
            num_confirmed_tracks=2,
            id_switches=0,
            mostly_tracked_ratio=1.00,
        ),
        "Full Fusion": EvaluationMetrics(
            rmse_position=2.15,
            rmse_velocity=0.45,
            num_confirmed_tracks=2,
            id_switches=0,
            mostly_tracked_ratio=1.00,
        ),
    }

    # Act
    report = evaluator.generate_markdown_report(results, scenario_name="scenario2")

    # Assert
    assert "# Báo Cáo Đánh Giá" in report
    assert "scenario2" in report
    assert "Radar-Only" in report
    assert "Radar + IR" in report
    assert "Full Fusion" in report
    assert "| 4.25" in report
    assert "100.0%" in report


def test_generate_markdown_report_empty_results_returns_placeholder() -> None:
    """Kiểm tra xuất báo cáo khi không có kết quả thử nghiệm."""
    # Arrange
    evaluator = NavalEvaluator()

    # Act
    report = evaluator.generate_markdown_report({}, scenario_name="scenario_empty")

    # Assert
    assert "scenario_empty" in report
    assert "Không có dữ liệu" in report


def test_evaluate_tracks_on_autoferry_scenario2_sample() -> None:
    """Kiểm tra tích hợp đánh giá trên kịch bản thực tế Autoferry scenario2."""
    # Arrange
    scenario_dir = Path("datasets/Autoferry-sensor_fusion_dataset/scenario2")
    if not scenario_dir.exists():
        pytest.skip("Autoferry scenario2 không tồn tại.")

    loader = AutoferryLoader()
    _, ground_truth = loader.load_scenario(scenario_dir)

    evaluator = NavalEvaluator()

    # Tạo các vết giả lập bám sát ground truth với nhiễu 0.2m
    confirmed_history: dict[float, list[TrackState]] = {}
    for step in ground_truth[:50]:
        step_dict_list: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], step)
            if isinstance(step, list)
            else [cast(dict[str, Any], step)]
        )
        step_times: list[float] = [float(it["time"]) for it in step_dict_list]
        step_time = float(np.mean(step_times))
        tracks: list[TrackState] = []
        for it in step_dict_list:
            pos_raw = cast(list[float], it["position"])
            target_id = int(it["targetID"])
            trk = TrackState(
                track_id=target_id,
                state_vector=np.array(
                    [pos_raw[0] + 0.2, pos_raw[1] - 0.2, 0.0, 0.0],
                    dtype=np.float64,
                ),
                covariance=np.eye(4, dtype=np.float64),
                last_updated=step_time,
                hits=5,
                misses=0,
                is_confirmed=True,
                contributing_sensors={SensorType.RADAR},
            )
            tracks.append(trk)
        confirmed_history[step_time] = tracks

    # Act
    metrics = evaluator.evaluate_tracks(confirmed_history, ground_truth[:50])

    # Assert
    # Sai số thực tế ~ sqrt(0.2^2 + 0.2^2) = ~0.2828m
    assert metrics.rmse_position < 1.0
    assert metrics.num_confirmed_tracks == 2
    assert metrics.id_switches == 0
    assert metrics.mostly_tracked_ratio == 1.0
