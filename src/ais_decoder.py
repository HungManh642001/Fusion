"""Mô-đun giải mã và ngoại suy động học thông điệp AIS đường thủy nội địa.

Mô-đun xử lý tệp văn bản chứa các bức điện chuẩn NMEA AIVDM/AIVDO gắn nhãn
thời gian, lọc bỏ dữ liệu nhiễu và ngoại suy vị trí trắc địa theo thời gian thực.
"""

import logging
import math
from dataclasses import dataclass

import pyais
from pyais.exceptions import AISBaseException

from src.config import (
    EARTH_RADIUS_M,
    KNOTS_TO_MS,
    LAT_BOUNDS_DEFAULT,
    LON_BOUNDS_DEFAULT,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AISMessage:
    """Biểu diễn bản tin AIS đường thủy đã được chuẩn hóa.

    Attributes:
        ts: Dấu thời gian tính theo mili-giây.
        mmsi: Mã nhận dạng dịch vụ di động hàng hải duy nhất của tàu thuyền.
        lat: Vĩ độ địa lý theo độ thập phân.
        lon: Kinh độ địa lý theo độ thập phân.
        sog: Vận tốc di chuyển trên mặt đất (Speed Over Ground) theo hải lý/giờ.
        cog: Hướng di chuyển trên mặt đất (Course Over Ground) theo độ (0 - 360).
    """

    ts: int
    mmsi: int
    lat: float
    lon: float
    sog: float
    cog: float


def extrapolate_kinematic(
    msg: AISMessage,
    *,
    target_ts: int,
) -> tuple[float, float]:
    """Ngoại suy tọa độ vĩ độ và kinh độ của tàu thuyền tới thời điểm mục tiêu.

    Áp dụng phương trình bình sai động học Dead-Reckoning (Công thức 8 và 9
    trong bài báo Sensors 2025, 25, 2179).

    Args:
        msg: Bản tin AIS cơ sở tại thời điểm thu nhận gần nhất.
        target_ts: Thời điểm mục tiêu cần ngoại suy tọa độ (mili-giây).

    Returns:
        Bộ giá trị (vĩ độ ngoại suy, kinh độ ngoại suy) theo độ thập phân.
    """
    dt_sec = (target_ts - msg.ts) / 1000.0
    if abs(dt_sec) < 1e-6 or math.isnan(msg.sog):
        return msg.lat, msg.lon

    sog_ms = msg.sog * KNOTS_TO_MS
    cog_rad = math.radians(msg.cog if not math.isnan(msg.cog) else 0.0)

    delta_lat = (180.0 * sog_ms * dt_sec * math.cos(cog_rad)) / (
        math.pi * EARTH_RADIUS_M
    )
    lat_target = msg.lat + delta_lat

    cos_lat = math.cos(math.radians(msg.lat))
    cos_lat_safe = 1e-6 if abs(cos_lat) < 1e-6 else cos_lat
    delta_lon = (180.0 * sog_ms * dt_sec * math.sin(cog_rad)) / (
        math.pi * EARTH_RADIUS_M * cos_lat_safe
    )
    lon_target = msg.lon + delta_lon

    return lat_target, lon_target


class AISDecoder:
    """Bộ giải mã thông điệp NMEA AIS cho khu vực giám sát luồng hàng hải."""

    def __init__(
        self,
        *,
        lat_bounds: tuple[float, float] = LAT_BOUNDS_DEFAULT,
        lon_bounds: tuple[float, float] = LON_BOUNDS_DEFAULT,
    ) -> None:
        """Khởi tạo bộ giải mã với giới hạn vùng địa lý hợp lệ.

        Args:
            lat_bounds: Khoảng vĩ độ hợp lệ (nhỏ nhất, lớn nhất).
            lon_bounds: Khoảng kinh độ hợp lệ (nhỏ nhất, lớn nhất).
        """
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

    def decode_file(self, filepath: str) -> list[AISMessage]:
        """Đọc và giải mã toàn bộ bản tin AIS từ tệp tin dữ liệu.

        Args:
            filepath: Đường dẫn tệp tin văn bản chứa nhật ký bản tin AIS.

        Returns:
            Danh sách các đối tượng AISMessage hợp lệ đã chuẩn hóa.

        Raises:
            FileNotFoundError: Khi không tìm thấy đường dẫn tệp tin AIS chỉ định.
        """
        messages: list[AISMessage] = []
        last_valid_cog: dict[int, float] = {}
        last_valid_sog: dict[int, float] = {}

        try:
            with open(filepath, encoding="utf-8", errors="ignore") as file:
                for line in file:
                    msg = self._process_line(line, last_valid_cog, last_valid_sog)
                    if msg is not None:
                        messages.append(msg)
        except FileNotFoundError as exc:
            logger.exception("Không tìm thấy tệp tin nhật ký AIS: %s", filepath)
            raise FileNotFoundError(f"Tệp AIS không tồn tại: {filepath}") from exc

        return messages

    def _process_line(
        self,
        line: str,
        last_valid_cog: dict[int, float],
        last_valid_sog: dict[int, float],
    ) -> AISMessage | None:
        """Xử lý và bóc tách một dòng văn bản thô chứa nhãn thời gian và bức điện.

        Args:
            line: Chuỗi văn bản một dòng trong tệp nhật ký.
            last_valid_cog: Bảng tra cứu hướng di chuyển hợp lệ gần nhất theo MMSI.
            last_valid_sog: Bảng tra cứu vận tốc hợp lệ gần nhất theo MMSI.

        Returns:
            Đối tượng AISMessage nếu giải mã thành công và nằm trong vùng bao,
            ngược lại trả về None.
        """
        stripped_line = line.strip()
        if not stripped_line:
            return None

        parts = stripped_line.split(":", 1)
        if len(parts) != 2:
            return None

        try:
            timestamp = int(parts[0])
        except ValueError:
            return None

        nmea_sentence = parts[1]
        if not (
            nmea_sentence.startswith("!AIVDM") or nmea_sentence.startswith("!AIVDO")
        ):
            return None

        return self._decode_sentence(
            timestamp=timestamp,
            nmea_sentence=nmea_sentence,
            last_valid_cog=last_valid_cog,
            last_valid_sog=last_valid_sog,
        )

    def _decode_sentence(
        self,
        *,
        timestamp: int,
        nmea_sentence: str,
        last_valid_cog: dict[int, float],
        last_valid_sog: dict[int, float],
    ) -> AISMessage | None:
        """Giải mã bức điện NMEA sang tọa độ và trạng thái động học.

        Args:
            timestamp: Dấu thời gian mili-giây của bức điện.
            nmea_sentence: Chuỗi bức điện NMEA AIVDM/AIVDO.
            last_valid_cog: Bảng tra cứu COG hợp lệ gần nhất theo MMSI.
            last_valid_sog: Bảng tra cứu SOG hợp lệ gần nhất theo MMSI.

        Returns:
            Bản tin AIS hợp lệ hoặc None nếu lỗi định dạng hoặc nằm ngoài vùng phủ.
        """
        try:
            decoded = pyais.decode(nmea_sentence)
            mmsi_val = getattr(decoded, "mmsi", None)
            lat_val = getattr(decoded, "lat", None)
            lon_val = getattr(decoded, "lon", None)
            sog_val = getattr(decoded, "speed", None)
            cog_val = getattr(decoded, "course", None)

            if mmsi_val is None or lat_val is None or lon_val is None:
                return None

            mmsi = int(mmsi_val)
            lat = float(lat_val)
            lon = float(lon_val)

            if not (self.lat_bounds[0] <= lat <= self.lat_bounds[1]):
                return None
            if not (self.lon_bounds[0] <= lon <= self.lon_bounds[1]):
                return None

            eff_cog = self._extract_effective_cog(mmsi, cog_val, last_valid_cog)
            eff_sog = self._extract_effective_sog(mmsi, sog_val, last_valid_sog)

            return AISMessage(
                ts=timestamp,
                mmsi=mmsi,
                lat=lat,
                lon=lon,
                sog=eff_sog,
                cog=eff_cog,
            )
        except (AISBaseException, ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _extract_effective_cog(
        mmsi: int,
        raw_cog: object,
        last_valid_cog: dict[int, float],
    ) -> float:
        """Trích xuất giá trị hướng di chuyển COG hiệu dụng, lọc giá trị đặc biệt 360.

        Args:
            mmsi: Mã định danh tàu thuyền.
            raw_cog: Giá trị COG thô bóc tách từ bức điện.
            last_valid_cog: Bảng tra cứu COG hợp lệ trước đó.

        Returns:
            Giá trị COG hiệu dụng theo độ từ 0.0 đến dưới 360.0.
        """
        if raw_cog is not None:
            try:
                val = float(raw_cog)  # type: ignore[arg-type]
                if 0.0 <= val < 360.0:
                    last_valid_cog[mmsi] = val
                    return val
            except (ValueError, TypeError):
                pass
        return last_valid_cog.get(mmsi, 0.0)

    @staticmethod
    def _extract_effective_sog(
        mmsi: int,
        raw_sog: object,
        last_valid_sog: dict[int, float],
    ) -> float:
        """Trích xuất giá trị vận tốc SOG hiệu dụng, lọc giá trị vượt ngưỡng 102 hải lý.

        Args:
            mmsi: Mã định danh tàu thuyền.
            raw_sog: Giá trị SOG thô bóc tách từ bức điện.
            last_valid_sog: Bảng tra cứu SOG hợp lệ trước đó.

        Returns:
            Giá trị SOG hiệu dụng theo hải lý/giờ.
        """
        if raw_sog is not None:
            try:
                val = float(raw_sog)  # type: ignore[arg-type]
                if 0.0 <= val < 102.0:
                    last_valid_sog[mmsi] = val
                    return val
            except (ValueError, TypeError):
                pass
        return last_valid_sog.get(mmsi, 0.0)
