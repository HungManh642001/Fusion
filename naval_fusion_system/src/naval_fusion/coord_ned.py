"""Các phép toán hình học và chuyển đổi tọa độ hệ Piren NED (North-East-Down).

Module cung cấp các hàm hỗ trợ bù chuyển động tàu mẹ (ownship motion compensation),
chuẩn hóa góc phương vị trong hệ cảm biến quang điện tử / hồng ngoại, và tính toán
góc phương vị giữa tàu mẹ và mục tiêu trong hệ tọa độ phẳng hàng hải.
"""

import math
from collections.abc import Sequence

import numpy as np

# Mốc trắc địa Piren (Trondheim, Na Uy) theo WGS-84: [Vĩ độ, Kinh độ, Độ cao]
PIREN_ORIGIN_LLA: tuple[float, float, float] = (63.4389029, 10.39908278, 39.923)


def compensate_ownship_2d(
    ownship_pos: tuple[float, float] | Sequence[float],
    vessel_meas: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Bù vector vị trí tàu mẹ vào điểm đo thân tàu để ra tọa độ NED thế giới.

    Thực hiện phép tịnh tiến tọa độ 2D:
        p_N = N_own + x_vessel
        p_E = E_own + y_vessel

    Args:
        ownship_pos: Tọa độ vị trí tàu mẹ (pos_north, pos_east) tính bằng mét.
        vessel_meas: Điểm đo 2D [x_vessel, y_vessel] trong hệ thân tàu (mét).

    Returns:
        np.ndarray: Vector tọa độ 2D [p_N, p_E] trong hệ Piren NED (dtype=float64).

    Raises:
        ValueError: Nếu ownship_pos hoặc vessel_meas không đúng 2 phần tử.
    """
    if len(ownship_pos) != 2:
        msg = (
            f"ownship_pos phải có đúng 2 phần tử (pos_north, pos_east), "
            f"nhận được {len(ownship_pos)}."
        )
        raise ValueError(msg)

    if len(vessel_meas) != 2:
        msg = (
            f"vessel_meas phải có đúng 2 phần tử [x_vessel, y_vessel], "
            f"nhận được {len(vessel_meas)}."
        )
        raise ValueError(msg)

    pos_n = float(ownship_pos[0]) + float(vessel_meas[0])
    pos_e = float(ownship_pos[1]) + float(vessel_meas[1])
    return np.array([pos_n, pos_e], dtype=np.float64)


def normalize_bearing_rad(angle_rad: float) -> float:
    """Chuẩn hóa góc phương vị về khoảng [-pi, pi] radian.

    Sử dụng hàm atan2(sin(theta), cos(theta)) để khử tính tuần hoàn
    và đưa góc về phạm vi đối xứng quanh 0 rad.

    Args:
        angle_rad: Góc phương vị đầu vào (radian).

    Returns:
        float: Góc phương vị đã chuẩn hóa trong khoảng [-pi, pi] radian.
    """
    return float(math.atan2(math.sin(angle_rad), math.cos(angle_rad)))


def calc_bearing_from_ownship(
    ownship_pos: tuple[float, float] | Sequence[float],
    target_pos_ned: np.ndarray | Sequence[float],
) -> float:
    """Tính góc phương vị (bearing) từ tàu mẹ tới mục tiêu trong hệ tọa độ Piren NED.

    Quy chuẩn hệ tọa độ hàng hải NED:
    - Hướng Bắc (True North, trục North): 0 rad.
    - Hướng Đông (True East, trục East): pi / 2 rad (+1.5708 rad).
    - Hướng Nam: pi rad.
    - Hướng Tây: -pi / 2 rad (-1.5708 rad).
    - Công thức lượng giác: theta = atan2(delta_East, delta_North).

    Args:
        ownship_pos: Tọa độ tàu mẹ (pos_north, pos_east) tính bằng mét.
        target_pos_ned: Tọa độ mục tiêu trong hệ NED, tối thiểu 2 phần tử [p_N, p_E].
            Hỗ trợ vector trạng thái động học 4D [p_N, p_E, v_N, v_E].

    Returns:
        float: Góc phương vị từ tàu mẹ tới mục tiêu trong khoảng [-pi, pi] radian.

    Raises:
        ValueError: Nếu ownship_pos không có đúng 2 phần tử hoặc target_pos_ned
            có ít hơn 2 phần tử.
    """
    if len(ownship_pos) != 2:
        msg = (
            f"ownship_pos phải có đúng 2 phần tử (pos_north, pos_east), "
            f"nhận được {len(ownship_pos)}."
        )
        raise ValueError(msg)

    if len(target_pos_ned) < 2:
        msg = (
            f"target_pos_ned phải có ít nhất 2 phần tử [p_N, p_E], "
            f"nhận được {len(target_pos_ned)}."
        )
        raise ValueError(msg)

    delta_north = float(target_pos_ned[0]) - float(ownship_pos[0])
    delta_east = float(target_pos_ned[1]) - float(ownship_pos[1])

    # atan2(delta_east, delta_north) tính góc từ trục North hướng về trục East
    raw_bearing = math.atan2(delta_east, delta_north)
    return normalize_bearing_rad(raw_bearing)
