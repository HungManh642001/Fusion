"""Mô-đun hợp nhất quỹ đạo đa nguồn theo cơ chế kết hợp tầng (Cascade Matching).

Triển khai thuật toán Cascade Association Matching (Sensors 2025, 25, 2179),
tính toán ma trận chi phí động học kết hợp khoảng cách không gian, phương vị
và độ tương đồng cosine của vector vận tốc trong không gian tọa độ radar quy chuẩn.
"""

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.optimize

from src.ais_decoder import AISMessage, extrapolate_kinematic
from src.config import FusionType
from src.coord_transform import CoordinateTransformer

logger = logging.getLogger(__name__)

_LSA_FUNC = cast(
    Callable[[object], tuple[Sequence[int], Sequence[int]]],
    scipy.optimize.linear_sum_assignment,
)


@dataclass(frozen=True)
class TrajectoryState:
    """Trạng thái động học của một phân đoạn quỹ đạo trong không gian tọa độ chung.

    Attributes:
        x: Hoành độ trong không gian radar chuẩn hóa.
        y: Tung độ trong không gian radar chuẩn hóa.
        vx: Vận tốc theo phương x.
        vy: Vận tốc theo phương y.
        azimuth: Góc phương vị chuyển động tính bằng radian.
        entry_time: Thời điểm xuất hiện đầu tiên của quỹ đạo (mili-giây).
        length: Chiều dài quỹ đạo (số điểm quan sát).
    """

    x: float
    y: float
    vx: float
    vy: float
    azimuth: float
    entry_time: int
    length: int


@dataclass(frozen=True)
class FusedTrack:
    """Biểu diễn đối tượng quỹ đạo sau khi được hợp nhất đa nguồn.

    Attributes:
        fusion_id: Mã định danh kết quả hợp nhất.
        fusion_type: Phân loại mức độ hợp nhất (tri, dual, single).
        mmsi: Mã nhận dạng AIS nếu có liên kết.
        radar_id: Mã định danh mục tiêu radar nếu có liên kết.
        camera_id: Mã định danh mục tiêu camera nếu có liên kết.
        camera_conf: Độ tin cậy quan sát từ camera.
        bbox_vis: Hộp bao mục tiêu trên ảnh camera [x, y, w, h] hoặc None.
        bbox_rad: Hộp bao mục tiêu trên ảnh radar [x, y, w, h] hoặc None.
        lat: Vĩ độ địa lý WGS-84 (độ thập phân).
        lon: Kinh độ địa lý WGS-84 (độ thập phân).
        sog: Vận tốc di chuyển mặt đất (hải lý/giờ).
        cog: Hướng di chuyển mặt đất (độ).
    """

    fusion_id: int
    fusion_type: FusionType
    mmsi: int | None
    radar_id: int | None
    camera_id: int | None
    camera_conf: float
    bbox_vis: list[float] | None
    bbox_rad: list[float] | None
    lat: float
    lon: float
    sog: float
    cog: float


class CascadeFusion:
    """Mô-đun hợp nhất kết hợp tầng đa nguồn AIS, Radar và Camera khả kiến."""

    def __init__(
        self,
        *,
        coord_transformer: CoordinateTransformer | None = None,
        match_cost_thresh: float = 3.0,
    ) -> None:
        """Khởi tạo mô-đun hợp nhất tầng.

        Args:
            coord_transformer: Bộ chuyển đổi hệ tọa độ giữa các cảm biến.
            match_cost_thresh: Ngưỡng chi phí tối đa để chấp nhận ghép nối quỹ đạo.
        """
        self.transformer: CoordinateTransformer = (
            coord_transformer
            if coord_transformer is not None
            else CoordinateTransformer()
        )
        self.match_cost_thresh: float = match_cost_thresh
        self.fusion_counter: int = 0
        self.radar_mmsi_history: dict[int, int] = {}

    def compute_cost(
        self,
        s1: TrajectoryState,
        s2: TrajectoryState,
        *,
        k: float = 0.5,
        l_limit: int = 100,
        t_limit: int = 60000,
    ) -> float:
        """Tính chi phí liên kết động học giữa hai trạng thái quỹ đạo (Eq. 12 & 13).

        Args:
            s1: Trạng thái động học của quỹ đạo nguồn 1.
            s2: Trạng thái động học của quỹ đạo nguồn 2.
            k: Hệ số cân bằng giữa khoảng cách hình học và sai lệch góc phương vị.
            l_limit: Giới hạn chênh lệch chiều dài quỹ đạo tối đa cho phép.
            t_limit: Giới hạn chênh lệch thời điểm bắt đầu tối đa cho phép (mili-giây).

        Returns:
            Giá trị chi phí liên kết (chi phí càng nhỏ độ tương đồng càng cao).
        """
        len_diff = abs(s1.length - s2.length)
        time_diff = abs(s1.entry_time - s2.entry_time)
        if len_diff > l_limit or time_diff > t_limit:
            return float("inf")

        dist = math.hypot(s1.x - s2.x, s1.y - s2.y)
        v1_mag = math.hypot(s1.vx, s1.vy)
        v2_mag = math.hypot(s2.vx, s2.vy)

        if v1_mag > 0.05 and v2_mag > 0.05:
            diff_v = abs(v1_mag - v2_mag)
            diff_az = abs(s1.azimuth - s2.azimuth)
            if diff_az > math.pi:
                diff_az = 2.0 * math.pi - diff_az
            diff_az_norm = diff_az / math.pi

            dot_prod = s1.vx * s2.vx + s1.vy * s2.vy
            eps = 1e-6
            cos_val = dot_prod / (v1_mag * v2_mag)
            cos_dist = (cos_val + 1.0 + eps) / 2.0
            cos_dist = max(eps, min(1.0, cos_dist))
        else:
            diff_v = 0.0
            diff_az_norm = 0.0
            cos_dist = 1.0

        cost = (k * dist + (1.0 - k) * diff_az_norm + diff_v) / cos_dist
        return float(cost)

    def fuse(
        self,
        ais_records: Sequence[Mapping[str, object]],
        radar_tracks: Sequence[Mapping[str, object]],
        visible_tracks: Sequence[Mapping[str, object]],
        *,
        ts: int,
    ) -> list[FusedTrack]:
        """Thực hiện 4 bước hợp nhất kết hợp tầng đa nguồn.

        Args:
            ais_records: Danh sách bản ghi AIS ngoại suy tại thời điểm ts.
            radar_tracks: Danh sách vết bám bắt từ radar.
            visible_tracks: Danh sách vết bám bắt từ camera khả kiến.
            ts: Dấu thời gian hiện tại (mili-giây).

        Returns:
            Danh sách các đối tượng FusedTrack đã được hợp nhất thông tin toàn diện.
        """
        ais_states = self._prepare_ais_states(ais_records, ts)
        rad_states = self._prepare_radar_states(radar_tracks, ts)
        vis_states = self._prepare_visible_states(visible_tracks, ts)

        # Bước 1: Liên kết AIS và Radar (k = 0.8)
        matched_ais_rad, unmatched_ais, _unmatched_rad = self._match_ais_radar(
            ais_states, rad_states
        )

        # Bước 2: Liên kết Radar và Visible dựa trên projection gate
        matched_rad_vis, unmatched_vis = self._match_radar_visible(
            rad_states, vis_states
        )

        # Bước 3: Liên kết AIS và Visible chưa được ghép cặp
        matched_ais_vis = self._match_ais_visible(
            ais_states, vis_states, unmatched_ais, unmatched_vis
        )

        # Bước 4: Tổng hợp các vết hợp nhất đa nguồn
        return self._synthesize_fused_tracks(
            matched_ais_rad=matched_ais_rad,
            matched_rad_vis=matched_rad_vis,
            matched_ais_vis=matched_ais_vis,
            unmatched_vis=unmatched_vis,
            ais_states=ais_states,
            rad_states=rad_states,
            vis_states=vis_states,
        )

    def _prepare_ais_states(
        self,
        ais_records: Sequence[Mapping[str, object]],
        ts: int,
    ) -> list[tuple[dict[str, object], TrajectoryState]]:
        """Quy đổi các bản ghi AIS về không gian trạng thái radar chuẩn hóa.

        Args:
            ais_records: Danh sách bản ghi AIS.
            ts: Dấu thời gian hiện tại.

        Returns:
            Danh sách cặp (bản ghi AIS gốc, TrajectoryState).
        """
        results: list[tuple[dict[str, object], TrajectoryState]] = []
        for a in ais_records:
            lat = float(cast(float | int | str, a["lat"]))
            lon = float(cast(float | int | str, a["lon"]))
            sog = float(cast(float | int | str, a.get("sog", 0.0)))
            cog = float(cast(float | int | str, a.get("cog", 0.0)))
            mmsi = int(cast(int | float | str, a["mmsi"]))
            xr, yr = self.transformer.geodetic_to_radar(lat, lon)

            temp_msg = AISMessage(
                ts=ts,
                mmsi=mmsi,
                lat=lat,
                lon=lon,
                sog=sog,
                cog=cog,
            )
            lat_next, lon_next = extrapolate_kinematic(temp_msg, target_ts=ts + 1000)
            xr_next, yr_next = self.transformer.geodetic_to_radar(lat_next, lon_next)
            vx_rad = xr_next - xr
            vy_rad = yr_next - yr
            v_mag_rad = math.hypot(vx_rad, vy_rad)
            az_rad = math.atan2(vx_rad, vy_rad) if v_mag_rad > 1e-4 else 0.0
            entry_t = int(cast(int | float | str, a.get("ts", ts)))
            state = TrajectoryState(
                x=xr / 2048.0,
                y=yr / 2048.0,
                vx=vx_rad / 20.0,
                vy=vy_rad / 20.0,
                azimuth=az_rad,
                entry_time=entry_t,
                length=10,
            )
            results.append((dict(a), state))
        return results

    def _prepare_radar_states(
        self,
        radar_tracks: Sequence[Mapping[str, object]],
        ts: int,
    ) -> list[tuple[dict[str, object], TrajectoryState]]:
        """Xây dựng trạng thái động học cho các vết radar trong không gian chuẩn hóa.

        Args:
            radar_tracks: Danh sách vết radar.
            ts: Dấu thời gian hiện tại.

        Returns:
            Danh sách cặp (vết radar gốc, TrajectoryState).
        """
        results: list[tuple[dict[str, object], TrajectoryState]] = []
        for r in radar_tracks:
            raw_box = cast(Sequence[object], r.get("bbox", []))
            bbox = [float(cast(float | int | str, x)) for x in raw_box]
            xc = bbox[0] + bbox[2] / 2.0 if len(bbox) == 4 else 0.0
            yc = bbox[1] + bbox[3] / 2.0 if len(bbox) == 4 else 0.0

            raw_traj = r.get("trajectory", [])
            traj: list[list[float]] = []
            if isinstance(raw_traj, Sequence) and not isinstance(
                raw_traj, (str, bytes)
            ):
                for pt in cast(Sequence[object], raw_traj):
                    if isinstance(pt, Sequence) and not isinstance(pt, (str, bytes)):
                        traj.append(
                            [
                                float(cast(float | int | str, coord))
                                for coord in cast(Sequence[object], pt)
                            ]
                        )

            if len(traj) >= 2:
                win_size = min(len(traj), 5)
                p_last = traj[-1]
                p_prev = traj[-win_size]
                dx = (p_last[0] + p_last[2] / 2.0) - (p_prev[0] + p_prev[2] / 2.0)
                dy = (p_last[1] + p_last[3] / 2.0) - (p_prev[1] + p_prev[3] / 2.0)
                dt_frames = win_size - 1
                v_mag = math.hypot(dx, dy) / dt_frames
                if v_mag > 0.05:
                    vx = dx / (dt_frames * 20.0)
                    vy = dy / (dt_frames * 20.0)
                    az = math.atan2(dx, dy)
                else:
                    vx, vy, az = 0.0, 0.0, 0.0
            else:
                vx, vy, az = 0.0, 0.0, 0.0

            entry_t = int(cast(int | float | str, r.get("ts", ts)))
            state = TrajectoryState(
                x=xc / 2048.0,
                y=yc / 2048.0,
                vx=vx,
                vy=vy,
                azimuth=az,
                entry_time=entry_t,
                length=len(traj) if traj else 10,
            )
            results.append((dict(r), state))
        return results

    def _prepare_visible_states(
        self,
        visible_tracks: Sequence[Mapping[str, object]],
        ts: int,
    ) -> list[tuple[dict[str, object], TrajectoryState]]:
        """Quy đổi vết camera về không gian tọa độ radar xấp xỉ chuẩn hóa.

        Args:
            visible_tracks: Danh sách vết camera.
            ts: Dấu thời gian hiện tại.

        Returns:
            Danh sách cặp (vết camera gốc, TrajectoryState).
        """
        results: list[tuple[dict[str, object], TrajectoryState]] = []
        for v in visible_tracks:
            raw_box = cast(Sequence[object], v.get("bbox", []))
            bbox = [float(cast(float | int | str, x)) for x in raw_box]
            xv_c = bbox[0] + bbox[2] / 2.0 if len(bbox) == 4 else 0.0

            xr_est = 2300.0 - (xv_c / 2560.0) * 800.0
            yr_est = 1150.0
            entry_t = int(cast(int | float | str, v.get("ts", ts)))
            state = TrajectoryState(
                x=xr_est / 2048.0,
                y=yr_est / 2048.0,
                vx=0.2,
                vy=0.2,
                azimuth=0.0,
                entry_time=entry_t,
                length=10,
            )
            results.append((dict(v), state))
        return results

    def _match_ais_radar(
        self,
        ais_states: list[tuple[dict[str, object], TrajectoryState]],
        radar_states: list[tuple[dict[str, object], TrajectoryState]],
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """Liên kết từng cặp quỹ đạo AIS và Radar bằng thuật toán Hungarian.

        Args:
            ais_states: Tập trạng thái động học AIS.
            radar_states: Tập trạng thái động học Radar.

        Returns:
            Bộ 3 (bảng ghép ais_idx -> rad_idx, ais chưa ghép, radar chưa ghép).
        """
        matched_ais_rad: dict[int, int] = {}
        unmatched_ais = list(range(len(ais_states)))
        unmatched_rad = list(range(len(radar_states)))

        if not ais_states or not radar_states:
            return matched_ais_rad, unmatched_ais, unmatched_rad

        cost_ar = np.zeros((len(ais_states), len(radar_states)), dtype=np.float32)
        for i, (a_data, s_a) in enumerate(ais_states):
            a_mmsi = int(cast(int | float | str, a_data["mmsi"]))
            for j, (r_data, s_r) in enumerate(radar_states):
                c = self.compute_cost(s_a, s_r, k=0.8)
                dist_px = math.hypot(s_a.x - s_r.x, s_a.y - s_r.y) * 2048.0
                r_tid = int(cast(int | float | str, r_data["track_id"]))
                if (
                    r_tid in self.radar_mmsi_history
                    and self.radar_mmsi_history[r_tid] == a_mmsi
                    and dist_px < 50.0
                ):
                    c -= 0.5
                cost_ar[i, j] = np.float32(c)

        row_ind_raw, col_ind_raw = _LSA_FUNC(cost_ar)
        for r, c in zip(row_ind_raw, col_ind_raw, strict=False):
            r_idx, c_idx = int(r), int(c)
            if float(cost_ar[r_idx, c_idx]) < self.match_cost_thresh:
                matched_ais_rad[r_idx] = c_idx
                r_tid = int(cast(int | float | str, radar_states[c_idx][0]["track_id"]))
                a_mmsi = int(cast(int | float | str, ais_states[r_idx][0]["mmsi"]))
                self.radar_mmsi_history[r_tid] = a_mmsi
                if r_idx in unmatched_ais:
                    unmatched_ais.remove(r_idx)
                if c_idx in unmatched_rad:
                    unmatched_rad.remove(c_idx)

        return matched_ais_rad, unmatched_ais, unmatched_rad

    def _match_radar_visible(
        self,
        radar_states: list[tuple[dict[str, object], TrajectoryState]],
        vis_states: list[tuple[dict[str, object], TrajectoryState]],
    ) -> tuple[dict[int, int], list[int]]:
        """Liên kết từng cặp quỹ đạo Radar và Visible dựa vào gating hình học.

        Args:
            radar_states: Tập trạng thái động học Radar.
            vis_states: Tập trạng thái động học Visible.

        Returns:
            Bộ đôi (bảng ghép rad_idx -> vis_idx, danh sách visible chưa liên kết).
        """
        matched_rad_vis: dict[int, int] = {}
        unmatched_vis = list(range(len(vis_states)))

        if not radar_states or not vis_states:
            return matched_rad_vis, unmatched_vis

        cost_rv = np.zeros((len(radar_states), len(vis_states)), dtype=np.float32)
        for i, (r_data, _) in enumerate(radar_states):
            raw_r_box = cast(Sequence[object], r_data.get("bbox", []))
            r_box = [float(cast(float | int | str, x)) for x in raw_r_box]
            v_gate = self.transformer.radar_bbox_to_visible_gate(r_box)
            gate_cx = v_gate[0] + v_gate[2] / 2.0
            gate_cy = v_gate[1] + v_gate[3] / 2.0
            for j, (v_data, _) in enumerate(vis_states):
                raw_v_box = cast(Sequence[object], v_data.get("bbox", []))
                v_box = [float(cast(float | int | str, x)) for x in raw_v_box]
                box_cx = v_box[0] + v_box[2] / 2.0
                box_cy = v_box[1] + v_box[3] / 2.0
                dist_x = abs(box_cx - gate_cx) / 2560.0
                dist_y = abs(box_cy - gate_cy) / 1440.0
                cost_rv[i, j] = np.float32(0.2 * dist_x + 0.8 * dist_y)

        row_ind_raw, col_ind_raw = _LSA_FUNC(cost_rv)
        for r, c in zip(row_ind_raw, col_ind_raw, strict=False):
            r_idx, c_idx = int(r), int(c)
            if float(cost_rv[r_idx, c_idx]) < 0.6:
                matched_rad_vis[r_idx] = c_idx
                if c_idx in unmatched_vis:
                    unmatched_vis.remove(c_idx)

        return matched_rad_vis, unmatched_vis

    def _match_ais_visible(
        self,
        ais_states: list[tuple[dict[str, object], TrajectoryState]],
        vis_states: list[tuple[dict[str, object], TrajectoryState]],
        unmatched_ais: list[int],
        unmatched_vis: list[int],
    ) -> dict[int, int]:
        """Liên kết các vết AIS và Visible chưa ghép cặp qua phép chiếu địa lý.

        Args:
            ais_states: Tập trạng thái động học AIS.
            vis_states: Tập trạng thái động học Visible.
            unmatched_ais: Danh sách chỉ số AIS chưa liên kết.
            unmatched_vis: Danh sách chỉ số Visible chưa liên kết.

        Returns:
            Bảng ánh xạ ais_idx -> vis_idx.
        """
        matched_ais_vis: dict[int, int] = {}
        for a_idx in list(unmatched_ais):
            a_data, _ = ais_states[a_idx]
            lat_a = float(cast(float | int | str, a_data["lat"]))
            lon_a = float(cast(float | int | str, a_data["lon"]))
            xv_proj, _ = self.transformer.geodetic_to_visible(lat_a, lon_a)
            for v_idx in list(unmatched_vis):
                raw_v_box = cast(Sequence[object], vis_states[v_idx][0].get("bbox", []))
                v_box = [float(cast(float | int | str, x)) for x in raw_v_box]
                box_cx = v_box[0] + v_box[2] / 2.0
                if abs(xv_proj - box_cx) < 300.0:
                    matched_ais_vis[a_idx] = v_idx
                    unmatched_ais.remove(a_idx)
                    unmatched_vis.remove(v_idx)
                    break
        return matched_ais_vis

    def _synthesize_fused_tracks(
        self,
        *,
        matched_ais_rad: dict[int, int],
        matched_rad_vis: dict[int, int],
        matched_ais_vis: dict[int, int],
        unmatched_vis: list[int],
        ais_states: list[tuple[dict[str, object], TrajectoryState]],
        rad_states: list[tuple[dict[str, object], TrajectoryState]],
        vis_states: list[tuple[dict[str, object], TrajectoryState]],
    ) -> list[FusedTrack]:
        """Tổng hợp kết quả liên kết thành các đối tượng FusedTrack chuẩn hóa.

        Args:
            matched_ais_rad: Bảng ánh xạ AIS sang Radar.
            matched_rad_vis: Bảng ánh xạ Radar sang Visible.
            matched_ais_vis: Bảng ánh xạ AIS sang Visible độc lập.
            unmatched_vis: Danh sách chỉ số Visible đơn nguồn.
            ais_states: Danh sách trạng thái AIS.
            rad_states: Danh sách trạng thái Radar.
            vis_states: Danh sách trạng thái Visible.

        Returns:
            Danh sách các đối tượng FusedTrack hoàn chỉnh.
        """
        fused_tracks: list[FusedTrack] = []
        assigned_rad: set[int] = set()
        assigned_vis: set[int] = set()
        assigned_ais: set[int] = set()

        # 4a. Hợp nhất ba nguồn (Tri-modal: AIS + Radar + Visible)
        for a_idx, r_idx in matched_ais_rad.items():
            if r_idx in matched_rad_vis:
                v_idx = matched_rad_vis[r_idx]
                a_data = ais_states[a_idx][0]
                r_data = rad_states[r_idx][0]
                v_data = vis_states[v_idx][0]

                raw_r_box = cast(Sequence[object], r_data.get("bbox", []))
                raw_v_box = cast(Sequence[object], v_data.get("bbox", []))
                r_bbox = [float(cast(float | int | str, x)) for x in raw_r_box]
                v_bbox = [float(cast(float | int | str, x)) for x in raw_v_box]

                self.fusion_counter += 1
                fused_tracks.append(
                    FusedTrack(
                        fusion_id=self.fusion_counter,
                        fusion_type=FusionType.TRI,
                        mmsi=int(cast(int | float | str, a_data["mmsi"])),
                        radar_id=int(cast(int | float | str, r_data["track_id"])),
                        camera_id=int(cast(int | float | str, v_data["track_id"])),
                        camera_conf=float(
                            cast(float | int | str, v_data.get("conf", 1.0))
                        ),
                        bbox_vis=v_bbox,
                        bbox_rad=r_bbox,
                        lat=float(cast(float | int | str, a_data["lat"])),
                        lon=float(cast(float | int | str, a_data["lon"])),
                        sog=float(cast(float | int | str, a_data.get("sog", 0.0))),
                        cog=float(cast(float | int | str, a_data.get("cog", 0.0))),
                    )
                )
                assigned_ais.add(a_idx)
                assigned_rad.add(r_idx)
                assigned_vis.add(v_idx)

        # 4b. Hợp nhất hai nguồn (Dual: Radar + Visible)
        for r_idx, v_idx in matched_rad_vis.items():
            if r_idx not in assigned_rad and v_idx not in assigned_vis:
                r_data = rad_states[r_idx][0]
                v_data = vis_states[v_idx][0]

                raw_r_box = cast(Sequence[object], r_data.get("bbox", []))
                raw_v_box = cast(Sequence[object], v_data.get("bbox", []))
                r_bbox = [float(cast(float | int | str, x)) for x in raw_r_box]
                v_bbox = [float(cast(float | int | str, x)) for x in raw_v_box]

                self.fusion_counter += 1
                fused_tracks.append(
                    FusedTrack(
                        fusion_id=self.fusion_counter,
                        fusion_type=FusionType.DUAL_RAD_VIS,
                        mmsi=None,
                        radar_id=int(cast(int | float | str, r_data["track_id"])),
                        camera_id=int(cast(int | float | str, v_data["track_id"])),
                        camera_conf=float(
                            cast(float | int | str, v_data.get("conf", 1.0))
                        ),
                        bbox_vis=v_bbox,
                        bbox_rad=r_bbox,
                        lat=0.0,
                        lon=0.0,
                        sog=0.0,
                        cog=0.0,
                    )
                )
                assigned_rad.add(r_idx)
                assigned_vis.add(v_idx)

        # 4c. Hợp nhất hai nguồn (Dual: AIS + Visible)
        for a_idx, v_idx in matched_ais_vis.items():
            if a_idx not in assigned_ais and v_idx not in assigned_vis:
                a_data = ais_states[a_idx][0]
                v_data = vis_states[v_idx][0]

                raw_v_box = cast(Sequence[object], v_data.get("bbox", []))
                v_bbox = [float(cast(float | int | str, x)) for x in raw_v_box]

                self.fusion_counter += 1
                fused_tracks.append(
                    FusedTrack(
                        fusion_id=self.fusion_counter,
                        fusion_type=FusionType.DUAL_AIS_VIS,
                        mmsi=int(cast(int | float | str, a_data["mmsi"])),
                        radar_id=None,
                        camera_id=int(cast(int | float | str, v_data["track_id"])),
                        camera_conf=float(
                            cast(float | int | str, v_data.get("conf", 1.0))
                        ),
                        bbox_vis=v_bbox,
                        bbox_rad=None,
                        lat=float(cast(float | int | str, a_data["lat"])),
                        lon=float(cast(float | int | str, a_data["lon"])),
                        sog=float(cast(float | int | str, a_data.get("sog", 0.0))),
                        cog=float(cast(float | int | str, a_data.get("cog", 0.0))),
                    )
                )
                assigned_ais.add(a_idx)
                assigned_vis.add(v_idx)

        # 4d. Đơn nguồn: Vết Visible chưa liên kết
        for v_idx in unmatched_vis:
            if v_idx not in assigned_vis:
                v_data = vis_states[v_idx][0]
                raw_v_box = cast(Sequence[object], v_data.get("bbox", []))
                v_bbox = [float(cast(float | int | str, x)) for x in raw_v_box]

                self.fusion_counter += 1
                fused_tracks.append(
                    FusedTrack(
                        fusion_id=self.fusion_counter,
                        fusion_type=FusionType.SINGLE_VIS,
                        mmsi=None,
                        radar_id=None,
                        camera_id=int(cast(int | float | str, v_data["track_id"])),
                        camera_conf=float(
                            cast(float | int | str, v_data.get("conf", 1.0))
                        ),
                        bbox_vis=v_bbox,
                        bbox_rad=None,
                        lat=0.0,
                        lon=0.0,
                        sog=0.0,
                        cog=0.0,
                    )
                )

        return fused_tracks
