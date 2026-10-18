"""Bộ nạp và tiền xử lý dữ liệu Autoferry cho phân hệ hợp nhất hải quân.

Module này chịu trách nhiệm nạp các tệp dữ liệu kịch bản JSON từ bộ chuẩn
Autoferry Benchmark Dataset (như scenario2), trích xuất và chuẩn hóa các điểm đo
dị thể (Radar, LiDAR, IR, EO), bù chuyển động tàu mẹ trong hệ tọa độ Piren NED,
và nạp dữ liệu vết chuẩn kiểm chứng (Ground Truth).
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from naval_fusion.coord_ned import compensate_ownship_2d, normalize_bearing_rad
from naval_fusion.models import Detection, SensorType

logger = logging.getLogger(__name__)


class AutoferryLoader:
    """Bộ nạp và tiền xử lý dữ liệu cảm biến đa phương thức Autoferry."""

    def load_scenario(
        self,
        scenario_dir: str | Path,
    ) -> tuple[list[Detection], list[Any]]:
        """Nạp dữ liệu kịch bản gồm các điểm đo cảm biến và Ground Truth từ thư mục.

        Tự động tìm kiếm các tệp `*_detections.json` và `*_groundTruth.json` bên trong
        thư mục kịch bản được chỉ định. Chuyển đổi toàn bộ điểm đo sang cấu trúc
        `Detection` chuẩn hóa và sắp xếp theo thứ tự thời gian tăng dần.

        Args:
            scenario_dir: Đường dẫn thư mục kịch bản (ví dụ: `scenario2`).

        Returns:
            tuple[list[Detection], list[Any]]: Tuple chứa:
                - Danh sách các đối tượng Detection đã bù NED và sắp xếp theo thời gian.
                - Danh sách dữ liệu Ground Truth nguyên bản từ tệp JSON.

        Raises:
            FileNotFoundError: Nếu thư mục kịch bản hoặc các tệp JSON không tồn tại.
        """
        dir_path = Path(scenario_dir)
        if not dir_path.is_dir():
            msg = f"Thư mục kịch bản không tồn tại: {dir_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        scen_name = dir_path.name

        # Tìm tệp detections JSON
        det_file = dir_path / f"{scen_name}_detections.json"
        if not det_file.is_file():
            candidates = sorted(dir_path.glob("*_detections.json"))
            if candidates:
                det_file = candidates[0]
            else:
                msg = f"Không tìm thấy tệp detections JSON trong thư mục {dir_path}."
                logger.error(msg)
                raise FileNotFoundError(msg)

        # Tìm tệp groundTruth JSON
        gt_file = dir_path / f"{scen_name}_groundTruth.json"
        if not gt_file.is_file():
            candidates = sorted(dir_path.glob("*_groundTruth.json"))
            if not candidates:
                candidates = sorted(dir_path.glob("*groundTruth*.json"))
            if candidates:
                gt_file = candidates[0]
            else:
                msg = f"Không tìm thấy tệp groundTruth JSON trong thư mục {dir_path}."
                logger.error(msg)
                raise FileNotFoundError(msg)

        detections = self.load_detections(det_file)
        ground_truth = self.load_ground_truth(gt_file)

        logger.info(
            "Đã nạp kịch bản %s thành công: %d detections, %d bước ground truth.",
            scen_name,
            len(detections),
            len(ground_truth),
        )
        return detections, ground_truth

    def load_detections(self, filepath: str | Path) -> list[Detection]:
        """Đọc và chuyển đổi tệp JSON chứa các điểm đo thành danh sách Detection.

        Args:
            filepath: Đường dẫn tới tệp *_detections.json.

        Returns:
            list[Detection]: Danh sách điểm đo đã chuẩn hóa và sắp xếp theo thời gian.

        Raises:
            FileNotFoundError: Nếu tệp không tồn tại.
        """
        path = Path(filepath)
        if not path.is_file():
            msg = f"Tệp detections không tồn tại: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with path.open("r", encoding="utf-8") as file:
            raw_data: list[dict[str, Any]] = json.load(file)

        return self.parse_detection_records(raw_data)

    def load_ground_truth(self, filepath: str | Path) -> list[Any]:
        """Đọc tệp JSON chứa dữ liệu Ground Truth.

        Args:
            filepath: Đường dẫn tới tệp *_groundTruth.json.

        Returns:
            list[Any]: Danh sách các bước thời gian và vị trí mục tiêu chuẩn.

        Raises:
            FileNotFoundError: Nếu tệp không tồn tại.
        """
        path = Path(filepath)
        if not path.is_file():
            msg = f"Tệp groundTruth không tồn tại: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with path.open("r", encoding="utf-8") as file:
            gt_data: list[Any] = json.load(file)

        return gt_data

    def parse_detection_records(
        self,
        records: Sequence[dict[str, Any]],
    ) -> list[Detection]:
        """Phân tích danh sách bản ghi thô thành đối tượng Detection chuẩn hóa.

        Args:
            records: Chuỗi các từ điển dữ liệu trích xuất từ JSON.

        Returns:
            list[Detection]: Danh sách Detection sắp xếp theo thời gian tăng dần.
        """
        all_detections: list[Detection] = []
        for record in records:
            parsed = self.parse_single_record(record)
            all_detections.extend(parsed)

        all_detections.sort(key=lambda d: d.timestamp)
        return all_detections

    def parse_single_record(self, record: dict[str, Any]) -> list[Detection]:
        """Phân tích một bản ghi JSON cảm biến thành các đối tượng Detection riêng lẻ.

        Xử lý đa dạng cấu trúc điểm đo:
        - Radar/LiDAR (2D): Tách ma trận 2xK hoặc danh sách Nx2 thành từng Detection
          có bù vị trí tàu mẹ trong hệ NED.
        - IR/EO (1D): Tách số thực đơn hoặc danh sách góc phương vị thành từng Detection
          có chuẩn hóa góc về [-pi, pi].
        - Bỏ qua các bản ghi có điểm đo rỗng hoặc None.

        Args:
            record: Từ điển chứa các trường sensorID, time, ownshipPosition,
                measurement.


        Returns:
            list[Detection]: Danh sách các điểm đo Detection sinh ra từ bản ghi.

        Raises:
            ValueError: Nếu thiếu trường bắt buộc, sensorID không hợp lệ,
                hoặc cấu trúc điểm đo không thể giải mã.
        """
        sensor_id = record.get("sensorID")
        if sensor_id is None:
            msg = "Bản ghi cảm biến thiếu trường bắt buộc 'sensorID'."
            logger.error(msg)
            raise ValueError(msg)

        try:
            sensor_type = SensorType(sensor_id)
        except ValueError as err:
            msg = f"Không nhận diện được sensorID: {sensor_id}."
            logger.error(msg)
            raise ValueError(msg) from err

        time_val = record.get("time")
        if time_val is None:
            msg = "Bản ghi cảm biến thiếu trường bắt buộc 'time'."
            logger.error(msg)
            raise ValueError(msg)
        timestamp = float(time_val)

        ownship_raw = record.get("ownshipPosition")
        if ownship_raw is None or len(ownship_raw) != 2:
            msg = (
                f"ownshipPosition phải có đúng 2 phần tử (pos_north, pos_east), "
                f"nhận được: {ownship_raw}."
            )
            logger.error(msg)
            raise ValueError(msg)
        ownship_pos = (float(ownship_raw[0]), float(ownship_raw[1]))

        raw_meas = record.get("measurement")
        if raw_meas is None:
            return []

        detections: list[Detection] = []

        if sensor_type in (SensorType.LIDAR, SensorType.RADAR):
            points = self._extract_cartesian_points(raw_meas)
            for x_val, y_val in points:
                raw_vector = [x_val, y_val]
                meas_ned = compensate_ownship_2d(ownship_pos, raw_vector)
                detections.append(
                    Detection(
                        sensor_type=sensor_type,
                        timestamp=timestamp,
                        ownship_pos=ownship_pos,
                        raw_measurement=raw_vector,
                        measurement_ned=meas_ned,
                    )
                )
        elif sensor_type in (SensorType.IR, SensorType.EO):
            angles = self._extract_bearing_angles(raw_meas)
            for angle in angles:
                norm_angle = normalize_bearing_rad(angle)
                raw_vector = [angle]
                meas_ned = np.array([norm_angle], dtype=np.float64)
                detections.append(
                    Detection(
                        sensor_type=sensor_type,
                        timestamp=timestamp,
                        ownship_pos=ownship_pos,
                        raw_measurement=raw_vector,
                        measurement_ned=meas_ned,
                    )
                )

        return detections

    @staticmethod
    def _extract_cartesian_points(
        measurement: object,
    ) -> list[tuple[float, float]]:
        """Trích xuất danh sách các điểm tọa độ Descartes 2D (x, y) từ dữ liệu đo thô.

        Args:
            measurement: Dữ liệu đo thô dạng list, tuple hoặc numpy.ndarray.

        Returns:
            list[tuple[float, float]]: Danh sách các tọa độ 2D (x, y) trong hệ thân tàu.

        Raises:
            ValueError: Nếu kích thước mảng hoặc số chiều không hợp lệ.
        """
        try:
            arr: np.ndarray = np.asarray(measurement, dtype=np.float64)
        except (ValueError, TypeError) as err:
            msg = f"Kích thước điểm đo Descartes không hợp lệ: {measurement}."
            raise ValueError(msg) from err

        if arr.size == 0:
            return []

        if arr.ndim == 1:
            if arr.shape[0] == 2:
                return [(float(arr[0]), float(arr[1]))]
            msg = (
                f"Kích thước điểm đo Descartes không hợp lệ: "
                f"mảng 1D có độ dài {arr.shape[0]}."
            )
            raise ValueError(msg)

        if arr.ndim == 2:
            # Định dạng MATLAB 2xK: row 0 là X, row 1 là Y
            if arr.shape[0] == 2:
                num_points = int(arr.shape[1])
                return [
                    (float(arr[0, k]), float(arr[1, k]))
                    for k in range(num_points)
                ]
            # Định dạng Kx2 (K > 2): mỗi hàng là một điểm [x, y]
            if arr.shape[1] == 2:
                num_points = int(arr.shape[0])
                return [
                    (float(arr[k, 0]), float(arr[k, 1]))
                    for k in range(num_points)
                ]
            msg = (
                f"Kích thước điểm đo Descartes không hợp lệ: "
                f"mảng 2D shape {arr.shape}."
            )
            raise ValueError(msg)

        msg = f"Kích thước điểm đo Descartes không hợp lệ: {measurement}."
        raise ValueError(msg)

    @staticmethod
    def _extract_bearing_angles(measurement: object) -> list[float]:
        """Trích xuất danh sách góc phương vị (radian) từ dữ liệu đo quang/hồng ngoại.

        Args:
            measurement: Góc đơn (float/int) hoặc danh sách/mảng các góc.

        Returns:
            list[float]: Danh sách góc phương vị (radian).

        Raises:
            ValueError: Nếu kiểu dữ liệu không thể chuyển đổi thành float.
        """
        try:
            arr: np.ndarray = np.asarray(measurement, dtype=np.float64)
        except (ValueError, TypeError) as err:
            msg = f"Định dạng góc phương vị không hợp lệ: {measurement}."
            raise ValueError(msg) from err

        if arr.size == 0:
            return []

        flat: np.ndarray = arr.ravel()
        return [float(val) for val in flat]

