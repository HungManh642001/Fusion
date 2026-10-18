"""Quản lý vòng đời vết mục tiêu theo tiêu chuẩn quân sự và quy tắc M/N.

Module này cung cấp lớp `TrackManager` đảm nhiệm toàn bộ vòng đời của các vết
mục tiêu hàng hải trong hệ quy chiếu Piren NED:
- Nguyên tắc quân sự bắt buộc: Cảm biến đo góc phương vị 1D (IR / EO) tuyệt đối
  không được tự ý khởi tạo vết mới nhằm ngăn ngừa sinh vết ma (ghost tracks).
  Chỉ các cảm biến đo 2D có đầy đủ thông tin cự ly (Radar, LiDAR) mới được phép
  tạo vết sơ bộ (tentative).
- Quy tắc M/N chuyển trạng thái: Vết nghi ngờ chuyển sang trạng thái khẳng định
  (CONFIRMED) khi đạt ít nhất M lần ghép nối thành công trong cửa sổ N chu kỳ.
- Quản lý quán tính (coasting) và xóa vết (DELETED): Vết bị loại bỏ khỏi hệ thống
  khi số lần trượt liên tiếp đạt ngưỡng tối đa (max_misses) hoặc thời gian trôi
  vượt quá ngưỡng cho phép (max_coasting_time).
"""

import logging
from typing import cast

import numpy as np

from naval_fusion.models import Detection, SensorType, TrackState

logger = logging.getLogger(__name__)

# Hằng số cấu hình mặc định theo quy chuẩn tác chiến hải quân
DEFAULT_CONFIRM_HITS: int = 3  # Số lần trúng M trong luật M/N
DEFAULT_WINDOW_SIZE: int = 5  # Kích thước cửa sổ N trong luật M/N
DEFAULT_MAX_MISSES: int = 5  # Số chu kỳ liên tiếp trượt trước khi xóa vết
DEFAULT_MAX_COASTING_TIME: float = 5.0  # Thời gian cho phép chạy quán tính (giây)
DEFAULT_INITIAL_POS_COV: float = 25.0  # Phương sai ban đầu vị trí (5.0m sigma)^2
DEFAULT_INITIAL_VEL_COV: float = 100.0  # Phương sai ban đầu vận tốc (10.0m/s sigma)^2


class TrackManager:
    """Bộ quản lý vòng đời vết mục tiêu chuẩn quân sự theo máy trạng thái M/N.

    Attributes:
        confirm_hits: Số lần trúng tối thiểu để khẳng định vết (M).
        window_size: Kích thước cửa sổ đánh giá (N).
        max_misses: Số bước quét trượt liên tiếp tối đa cho phép.
        max_coasting_time: Thời gian tối đa vết được chạy quán tính (giây).
        initial_pos_cov: Phương sai ban đầu gán cho vị trí [m^2].
        initial_vel_cov: Phương sai ban đầu gán cho vận tốc [m^2/s^2].
    """

    def __init__(
        self,
        confirm_hits: int = DEFAULT_CONFIRM_HITS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_misses: int = DEFAULT_MAX_MISSES,
        max_coasting_time: float = DEFAULT_MAX_COASTING_TIME,
        initial_pos_cov: float = DEFAULT_INITIAL_POS_COV,
        initial_vel_cov: float = DEFAULT_INITIAL_VEL_COV,
    ) -> None:
        """Khởi tạo bộ quản lý vòng đời vết mục tiêu.

        Args:
            confirm_hits: Số lần ghép nối thành công để xác nhận vết (M >= 1).
            window_size: Chiều dài cửa sổ trượt M/N (N >= confirm_hits).
            max_misses: Số lần mất liên lạc liên tiếp tối đa trước khi xóa (>= 1).
            max_coasting_time: Thời gian trôi tối đa cho phép (giây > 0.0).
            initial_pos_cov: Phương sai vị trí ban đầu (m^2 > 0.0).
            initial_vel_cov: Phương sai vận tốc ban đầu (m^2/s^2 > 0.0).

        Raises:
            ValueError: Nếu bất kỳ tham số cấu hình nào nằm ngoài miền hợp lệ.
        """
        if confirm_hits < 1:
            msg = f"confirm_hits phải lớn hơn hoặc bằng 1, nhận được: {confirm_hits}"
            logger.error(msg)
            raise ValueError(msg)

        if window_size < confirm_hits:
            msg = (
                f"window_size ({window_size}) không được nhỏ hơn "
                f"confirm_hits ({confirm_hits})"
            )
            logger.error(msg)
            raise ValueError(msg)

        if max_misses < 1:
            msg = f"max_misses phải lớn hơn hoặc bằng 1, nhận được: {max_misses}"
            logger.error(msg)
            raise ValueError(msg)

        if max_coasting_time <= 0.0:
            msg = f"max_coasting_time phải là số dương, nhận được: {max_coasting_time}"
            logger.error(msg)
            raise ValueError(msg)

        if initial_pos_cov <= 0.0:
            msg = f"initial_pos_cov phải là số dương, nhận được: {initial_pos_cov}"
            logger.error(msg)
            raise ValueError(msg)

        if initial_vel_cov <= 0.0:
            msg = f"initial_vel_cov phải là số dương, nhận được: {initial_vel_cov}"
            logger.error(msg)
            raise ValueError(msg)

        self.confirm_hits: int = confirm_hits
        self.window_size: int = window_size
        self.max_misses: int = max_misses
        self.max_coasting_time: float = float(max_coasting_time)
        self.initial_pos_cov: float = float(initial_pos_cov)
        self.initial_vel_cov: float = float(initial_vel_cov)

        self._next_track_id: int = 1
        self._track_history: dict[int, list[bool]] = {}

    @property
    def next_track_id(self) -> int:
        """Định danh số nguyên kế tiếp sẽ được cấp phát cho vết mới."""
        return self._next_track_id

    def reset(self, initial_track_id: int = 1) -> None:
        """Đặt lại bộ đếm ID vết và xóa toàn bộ lịch sử trượt/trúng.

        Args:
            initial_track_id: Định danh vết bắt đầu sau khi reset (mặc định 1).
        """
        self._next_track_id = initial_track_id
        self._track_history.clear()
        logger.info(
            "TrackManager đã được đặt lại, next_track_id bắt đầu từ %d",
            initial_track_id,
        )

    def init_track(self, detection: Detection) -> TrackState | None:
        """Khởi tạo vết nghi ngờ (tentative) mới từ điểm đo cảm biến.

        Tuân thủ nghiêm ngặt nguyên tắc tác chiến hải quân: Tuyệt đối không cho phép
        khởi tạo vết mới từ các cảm biến đo góc 1D (IR hoặc EO) do thiếu thông tin
        cự ly, nhằm loại trừ hoàn toàn hiện tượng tạo vết ma (ghost tracks).

        Args:
            detection: Điểm đo ứng viên để khởi tạo vết.

        Returns:
            TrackState mới với is_confirmed=False nếu cảm biến là Radar hoặc LiDAR;
            trả về None nếu cảm biến là tia góc 1D hoặc không hợp lệ.
        """
        # NGUYÊN TẮC QUÂN SỰ BẮT BUỘC: Tia góc 1D không được tự ý tạo vết
        if detection.sensor_type in (SensorType.IR, SensorType.EO):
            logger.debug(
                "Từ chối khởi tạo vết từ cảm biến tia góc 1D %s theo quy chuẩn quân sự",
                detection.sensor_type.name,
            )
            return None

        # Chỉ chấp nhận cảm biến đo không gian 2D có thông tin cự ly (Radar, LiDAR)
        if detection.sensor_type not in (SensorType.RADAR, SensorType.LIDAR):
            logger.warning(
                "Cảm biến %s không được hỗ trợ khởi tạo vết",
                detection.sensor_type,
            )
            return None

        # Kiểm tra tính toàn vẹn của tọa độ đo bù NED
        if detection.measurement_ned.shape[0] < 2:
            logger.warning(
                "Tọa độ đo measurement_ned có kích thước không hợp lệ (< 2 phần tử)"
            )
            return None

        pos_north = float(detection.measurement_ned[0])
        pos_east = float(detection.measurement_ned[1])

        # Vector trạng thái ban đầu [pN, pE, vN=0.0, vE=0.0]
        state_vec = np.array([pos_north, pos_east, 0.0, 0.0], dtype=np.float64)

        # Ma trận hiệp phương sai ban đầu
        cov_matrix = np.diag(
            [
                self.initial_pos_cov,
                self.initial_pos_cov,
                self.initial_vel_cov,
                self.initial_vel_cov,
            ]
        ).astype(np.float64)

        track_id = self._next_track_id
        self._next_track_id += 1

        # Ghi nhận lần trúng đầu tiên vào lịch sử cửa sổ M/N
        self._track_history[track_id] = [True]

        new_track = TrackState(
            track_id=track_id,
            state_vector=state_vec,
            covariance=cov_matrix,
            last_updated=float(detection.timestamp),
            hits=1,
            misses=0,
            is_confirmed=False,
            contributing_sensors={detection.sensor_type},
        )

        logger.info(
            "Khởi tạo vết nghi ngờ track_id=%d từ cảm biến %s tại (pN=%.2f, pE=%.2f)",
            track_id,
            detection.sensor_type.name,
            pos_north,
            pos_east,
        )
        return new_track

    def update_matches(
        self,
        tracks: list[TrackState],
        matches: list[tuple[int, int]],
        detections: list[Detection],
        current_time: float,
    ) -> list[TrackState]:
        """Cập nhật các vết được ghép nối thành công với điểm đo cảm biến.

        Thực hiện tăng số lần trúng (`hits += 1`), đặt lại `misses = 0`, cập nhật
        mốc thời gian, bổ sung loại cảm biến đóng góp và thăng cấp trạng thái
        `is_confirmed = True` khi thỏa mãn quy tắc M/N.

        Args:
            tracks: Danh sách vết hiện tại.
            matches: Danh sách các cặp chỉ số (chỉ_số_vết, chỉ_số_điểm_đo).
            detections: Danh sách điểm đo tại chu kỳ hiện tại.
            current_time: Thời điểm hiện tại (giây).

        Returns:
            Danh sách vết sau khi đã được cập nhật dữ liệu ghép nối.
        """
        for track_idx, det_idx in matches:
            if not (0 <= track_idx < len(tracks) and 0 <= det_idx < len(detections)):
                logger.warning(
                    "Cặp ghép không hợp lệ (track_idx=%d, det_idx=%d)",
                    track_idx,
                    det_idx,
                )
                continue

            track = tracks[track_idx]
            det = detections[det_idx]

            track.hits += 1
            track.misses = 0
            track.last_updated = float(current_time)
            track.contributing_sensors.add(det.sensor_type)

            # Cập nhật lịch sử cửa sổ trượt M/N
            hist = self._track_history.setdefault(track.track_id, [])
            hist.append(True)
            if len(hist) > self.window_size:
                hist.pop(0)

            # Thăng cấp vết lên CONFIRMED nếu đạt đủ số lần trúng
            if not track.is_confirmed:
                hits_in_window = sum(hist)
                if (
                    track.hits >= self.confirm_hits
                    or hits_in_window >= self.confirm_hits
                ):
                    track.is_confirmed = True
                    logger.info(
                        "Vết track_id=%d đã được XÁC NHẬN với %d lần trúng "
                        "(%d/%d trong cửa sổ)",
                        track.track_id,
                        track.hits,
                        hits_in_window,
                        len(hist),
                    )

        return tracks

    def update_unmatched(
        self,
        tracks: list[TrackState],
        unmatched_track_indices: list[int],
        current_time: float,
    ) -> list[TrackState]:
        """Xử lý các vết không có điểm đo ghép nối và loại bỏ các vết chết.

        Tăng `misses += 1` cho các vết trượt. Nếu một vết vượt ngưỡng `max_misses`
        hoặc thời gian trôi kể từ lần cập nhật cuối vượt quá `max_coasting_time`,
        vết đó sẽ bị loại bỏ khỏi danh sách theo dõi.

        Args:
            tracks: Danh sách vết hiện tại.
            unmatched_track_indices: Danh sách chỉ số của các vết không được ghép.
            current_time: Thời điểm hiện tại (giây).

        Returns:
            Danh sách các vết còn sống sau khi đã thanh lọc vết chết.
        """
        for idx in unmatched_track_indices:
            if not (0 <= idx < len(tracks)):
                logger.warning("Chỉ số vết trượt không hợp lệ: %d", idx)
                continue

            track = tracks[idx]
            track.misses += 1

            # Ghi nhận lần trượt vào lịch sử cửa sổ
            hist = self._track_history.setdefault(track.track_id, [])
            hist.append(False)
            if len(hist) > self.window_size:
                hist.pop(0)

        # Thanh lọc vết chết: giữ lại các vết chưa vượt ngưỡng trượt và trôi
        surviving_tracks: list[TrackState] = []
        for track in tracks:
            if self._is_dead(track, current_time):
                coasting_time = current_time - track.last_updated
                logger.info(
                    "Loại bỏ vết chết track_id=%d: misses=%d/%d, "
                    "coasting_time=%.2fs/%.2fs",
                    track.track_id,
                    track.misses,
                    self.max_misses,
                    coasting_time,
                    self.max_coasting_time,
                )
                self._track_history.pop(track.track_id, None)
            else:
                surviving_tracks.append(track)

        return surviving_tracks

    def _is_dead(self, track: TrackState, current_time: float) -> bool:
        """Kiểm tra điều kiện tử vong của một vết mục tiêu.

        Args:
            track: Vết mục tiêu cần kiểm tra.
            current_time: Thời điểm hiện tại (giây).

        Returns:
            True nếu vết đã chết cần xóa bỏ; False nếu vết còn hợp lệ.
        """
        if track.misses >= self.max_misses:
            return True

        coasting_time = current_time - track.last_updated
        return coasting_time > self.max_coasting_time

    def step(
        self,
        tracks: list[TrackState],
        matches: list[tuple[int, int]],
        unmatched_tracks: list[int] | list[Detection] | None = None,
        unmatched_detections: list[int] | float | None = None,
        detections: list[Detection] | None = None,
        current_time: float | None = None,
    ) -> list[TrackState]:
        """Thực hiện toàn bộ chu trình quản lý vết trong một chu kỳ đo lường.

        Hỗ trợ 2 quy ước gọi:
        1. 6 tham số: `step(tracks, matches, unmatched_tracks, ...)`
        2. 4 tham số: `step(tracks, matches, detections, current_time)`

        Quy trình xử lý:
        1. Cập nhật các vết được ghép nối thành công (`update_matches`).
        2. Tăng số lần trượt và loại bỏ các vết chết (`update_unmatched`).
        3. Khởi tạo vết sơ bộ mới từ các điểm đo chưa ghép hợp lệ (`init_track`).
        4. Trả về danh sách tất cả các vết còn tồn tại.

        Args:
            tracks: Danh sách vết hiện tại.
            matches: Danh sách các cặp ghép (track_idx, det_idx).
            unmatched_tracks: Danh sách chỉ số vết không ghép, hoặc detections.
            unmatched_detections: Danh sách chỉ số đo không ghép, hoặc current_time.
            detections: Danh sách điểm đo (khi gọi 6 tham số).
            current_time: Thời điểm hiện tại (khi gọi 6 tham số).

        Returns:
            Danh sách các vết còn sống sau chu kỳ cập nhật.
        """
        # Phân biệt cú pháp gọi 4 tham số vs 6 tham số
        if isinstance(unmatched_detections, (int, float)):
            # Cú pháp rút gọn 4 tham số: step(tracks, matches, detections, current_time)
            actual_detections = cast(list[Detection], unmatched_tracks or [])
            actual_current_time = float(unmatched_detections)

            matched_trk_set = {m[0] for m in matches}
            actual_unmatched_tracks = [
                i for i in range(len(tracks)) if i not in matched_trk_set
            ]

            matched_det_set = {m[1] for m in matches}
            actual_unmatched_detections = [
                j for j in range(len(actual_detections)) if j not in matched_det_set
            ]
        else:
            # Cú pháp đầy đủ 6 tham số
            actual_detections = detections or []
            actual_current_time = (
                float(current_time) if current_time is not None else 0.0
            )

            if unmatched_tracks is not None:
                actual_unmatched_tracks = cast(list[int], unmatched_tracks)
            else:
                matched_trk_set = {m[0] for m in matches}
                actual_unmatched_tracks = [
                    i for i in range(len(tracks)) if i not in matched_trk_set
                ]

            if unmatched_detections is not None:
                actual_unmatched_detections = unmatched_detections
            else:
                matched_det_set = {m[1] for m in matches}
                actual_unmatched_detections = [
                    j for j in range(len(actual_detections)) if j not in matched_det_set
                ]

        # Bước 1: Cập nhật vết trúng
        working_tracks = list(tracks)
        working_tracks = self.update_matches(
            working_tracks,
            matches,
            actual_detections,
            actual_current_time,
        )

        # Bước 2: Cập nhật vết trượt và thanh lọc vết chết
        surviving_tracks = self.update_unmatched(
            working_tracks,
            actual_unmatched_tracks,
            actual_current_time,
        )

        # Bước 3: Sinh vết nghi ngờ mới từ các điểm đo chưa ghép hợp lệ
        for det_idx in actual_unmatched_detections:
            if 0 <= det_idx < len(actual_detections):
                new_track = self.init_track(actual_detections[det_idx])
                if new_track is not None:
                    surviving_tracks.append(new_track)

        return surviving_tracks

    def get_confirmed_tracks(self, tracks: list[TrackState]) -> list[TrackState]:
        """Lọc và trả về danh sách các vết đã được xác nhận (CONFIRMED).

        Args:
            tracks: Danh sách vết cần lọc.

        Returns:
            Danh sách chỉ chứa các vết có is_confirmed=True.
        """
        return [t for t in tracks if t.is_confirmed]

    def get_tentative_tracks(self, tracks: list[TrackState]) -> list[TrackState]:
        """Lọc và trả về danh sách các vết đang ở trạng thái nghi ngờ (TENTATIVE).

        Args:
            tracks: Danh sách vết cần lọc.

        Returns:
            Danh sách chỉ chứa các vết có is_confirmed=False.
        """
        return [t for t in tracks if not t.is_confirmed]
