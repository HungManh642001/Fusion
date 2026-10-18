"""Extended Object Tracking (EOT) Offset Compensator & Hydrodynamic Smoother.

Computes oriented bounding ellipse offsets to project radar/lidar hull surface
reflections back to the central RTK GNSS mast antenna, and provides hydrodynamic
Savitzky-Golay trajectory smoothing and WGS-84 geodetic conversion.
"""

from typing import Tuple
import numpy as np
from scipy.signal import savgol_filter


class EOTCompensator:
    """Compensates extended hull glint and computes smoothed maritime telemetry."""

    def __init__(
        self,
        lat0: float = 63.4389029,
        lon0: float = 10.3990828,
    ) -> None:
        """Initialize the EOT compensator with local Piren NED geodetic origin.

        Args:
            lat0: Latitude of Piren origin (degrees North).
            lon0: Longitude of Piren origin (degrees East).
        """
        self.lat0 = lat0
        self.lon0 = lon0
        self.m_per_deg_lat = 111319.5
        self.m_per_deg_lon = 111319.5 * np.cos(np.radians(lat0))

    def compute_antenna_offset(
        self,
        own_pos: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        length: float = 31.25,
        beam: float = 9.6,
    ) -> np.ndarray:
        """Compute the dynamic 2D offset vector from illuminated hull reflection to mast antenna.

        Args:
            own_pos: Current ownship position [North, East].
            target_pos: Estimated target center position [North, East].
            target_vel: Estimated target velocity vector [v_N, v_E].
            length: Vessel overall length in meters (default 31.25 for Gunnerus).
            beam: Vessel beam width in meters (default 9.6 for Gunnerus).

        Returns:
            np.ndarray: 2D offset vector [delta_N, delta_E] in meters.
        """
        speed = float(np.hypot(target_vel[0], target_vel[1]))
        if speed < 0.2:
            return np.array([0.0, 0.0], dtype=float)

        psi = np.arctan2(target_vel[1], target_vel[0])  # Heading COG relative to North
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        # Line of sight from ownship to target
        los = target_pos - own_pos
        los_norm = float(np.linalg.norm(los))
        if los_norm < 1.0:
            return np.array([0.0, 0.0], dtype=float)
        u_los = los / los_norm

        # Project LOS into vessel body frame (x forward, y starboard)
        los_body_x = c_psi * u_los[0] + s_psi * u_los[1]
        los_body_y = -s_psi * u_los[0] + c_psi * u_los[1]

        # Sensor hits the facing exterior hull edge
        # Offset to center mast is directed inwards (opposite to facing direction)
        delta_body_x = -float(np.sign(los_body_x)) * min(length / 4.0, 7.8)
        delta_body_y = -float(np.sign(los_body_y)) * min(beam / 4.0, 2.4)

        # Rotate body offset back into NED frame
        delta_ned_n = c_psi * delta_body_x - s_psi * delta_body_y
        delta_ned_e = s_psi * delta_body_x + c_psi * delta_body_y

        return np.array([delta_ned_n, delta_ned_e], dtype=float)

    def smooth_trajectory(
        self,
        trajectory: np.ndarray,
        window_len: int = 51,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Apply Savitzky-Golay hydrodynamic polynomial smoothing to 2D trajectory.

        Args:
            trajectory: (N, 2) array of coordinates.
            window_len: Target window length (odd integer).
            polyorder: Polynomial order (typically 2).

        Returns:
            np.ndarray: (N, 2) smoothed trajectory.
        """
        n_pts = len(trajectory)
        if n_pts <= polyorder + 2:
            return trajectory.copy()

        # Adapt window length to data length
        w = min(window_len, n_pts if n_pts % 2 == 1 else n_pts - 1)
        if w <= polyorder:
            w = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

        s_n = savgol_filter(trajectory[:, 0], w, polyorder)
        s_e = savgol_filter(trajectory[:, 1], w, polyorder)
        return np.column_stack([s_n, s_e])

    def to_wgs84_telemetry(
        self,
        pos_ned: np.ndarray,
        vel_ned: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Convert local NED position and velocity to WGS-84 Latitude, Longitude, SOG, COG.

        Args:
            pos_ned: [North, East] in meters.
            vel_ned: [v_N, v_E] in m/s.

        Returns:
            Tuple[float, float, float, float]: (lat, lon, sog_knots, cog_degrees)
        """
        lat = float(self.lat0 + pos_ned[0] / self.m_per_deg_lat)
        lon = float(self.lon0 + pos_ned[1] / self.m_per_deg_lon)
        sog = float(np.hypot(vel_ned[0], vel_ned[1]) * 1.94384)  # m/s to knots
        cog = float((np.degrees(np.arctan2(vel_ned[1], vel_ned[0])) + 360.0) % 360.0)
        return lat, lon, sog, cog
