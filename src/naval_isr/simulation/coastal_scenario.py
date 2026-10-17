"""Coastal Tactical Multi-Sensor Scenario Generator.

Simulates 3 tactical maritime vessels:
1. Cooperative Cargo vessel (AIS broadcasting, compliant trajectory)
2. AIS-Spoofed Smuggler/Spy vessel (AIS broadcasting false position offset)
3. Dark Warship (No AIS, high-G coordinated turn maneuver, active military radar emitter)

Sensors modeled:
- Active Coastal Radar: [range, azimuth, range_rate] with noise
- Passive ESM Station: [bearing, emitter_id] (bearing-only)
- AIS Base Station: periodic broadcast [MMSI, lat, lon, SOG, COG]
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np


@dataclass
class TacticalVessel:
    track_id: int
    vessel_type: str
    length_m: float
    has_ais: bool
    ais_spoofed: bool
    emitter_active: bool
    emitter_id: str
    mmsi: int
    trajectory: np.ndarray  # Shape: (N, 5) -> [t, x, y, vx, vy]


class CoastalScenarioSimulator:
    """Simulates realistic coastal surface surveillance scenarios with multiple sensors."""

    def __init__(
        self,
        origin_lat: float = 30.563549,
        origin_lon: float = 114.306087,
        radar_range_std: float = 15.0,  # meters
        radar_azimuth_std: float = 0.5 * np.pi / 180.0,  # ~0.5 deg in rad
        radar_rr_std: float = 0.5,  # m/s
        esm_bearing_std: float = 1.0 * np.pi / 180.0,  # ~1.0 deg in rad
        ais_period_sec: float = 6.0,
        random_seed: int = 42,
    ):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.radar_range_std = radar_range_std
        self.radar_azimuth_std = radar_azimuth_std
        self.radar_rr_std = radar_rr_std
        self.esm_bearing_std = esm_bearing_std
        self.ais_period_sec = ais_period_sec
        self.rng = np.random.default_rng(random_seed)
        self.R_earth = 6371000.0

    def enu_to_geodetic(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local East-North (meters) to Lat-Lon (degrees)."""
        lat0_rad = np.radians(self.origin_lat)
        d_lat = (y / self.R_earth) * (180.0 / np.pi)
        d_lon = (x / (self.R_earth * np.cos(lat0_rad))) * (180.0 / np.pi)
        return self.origin_lat + d_lat, self.origin_lon + d_lon

    def geodetic_to_enu(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert Lat-Lon (degrees) to local East-North (meters)."""
        lat0_rad = np.radians(self.origin_lat)
        d_lat = np.radians(lat - self.origin_lat)
        d_lon = np.radians(lon - self.origin_lon)
        y = d_lat * self.R_earth
        x = d_lon * self.R_earth * np.cos(lat0_rad)
        return float(x), float(y)

    def generate_scenario(
        self, duration_sec: int = 300, dt: float = 2.0
    ) -> Tuple[List[TacticalVessel], Dict[str, List[Any]]]:
        """Generate ground-truth vessel kinematics and synchronized multi-sensor feeds."""
        time_steps = np.arange(0.0, float(duration_sec), dt)
        num_steps = len(time_steps)

        # -------------------------------------------------------------
        # 1. Vessel 1: Cooperative Merchant / Cargo
        # -------------------------------------------------------------
        # Linear transit: [-15000, 20000] moving southeast [10, -3]
        x1_0, y1_0 = -15000.0, 20000.0
        vx1, vy1 = 10.0, -3.0
        traj1 = np.zeros((num_steps, 5))
        for i, t in enumerate(time_steps):
            traj1[i] = [t, x1_0 + vx1 * t, y1_0 + vy1 * t, vx1, vy1]

        vessel1 = TacticalVessel(
            track_id=1,
            vessel_type="cooperative",
            length_m=180.0,
            has_ais=True,
            ais_spoofed=False,
            emitter_active=False,
            emitter_id="NONE",
            mmsi=413806072,
            trajectory=traj1,
        )

        # -------------------------------------------------------------
        # 2. Vessel 2: AIS-Spoofed Target (Smuggler / Spy vessel)
        # -------------------------------------------------------------
        # Linear transit: [10000, 15000] moving northwest [-5, 8]
        # Spoofs AIS by broadcasting +3500m X, +2500m Y false location
        x2_0, y2_0 = 10000.0, 15000.0
        vx2, vy2 = -5.0, 8.0
        traj2 = np.zeros((num_steps, 5))
        for i, t in enumerate(time_steps):
            traj2[i] = [t, x2_0 + vx2 * t, y2_0 + vy2 * t, vx2, vy2]

        vessel2 = TacticalVessel(
            track_id=2,
            vessel_type="ais_spoofed",
            length_m=48.0,
            has_ais=True,
            ais_spoofed=True,
            emitter_active=False,
            emitter_id="NONE",
            mmsi=413999888,
            trajectory=traj2,
        )

        # -------------------------------------------------------------
        # 3. Vessel 3: Dark Warship (Guided Missile Destroyer)
        # -------------------------------------------------------------
        # Starts at [-5000, 30000], speed ~17 m/s, performs coordinated turn at t in [t_turn_start, t_turn_end]
        traj3 = np.zeros((num_steps, 5))
        curr_x, curr_y = -5000.0, 30000.0
        speed = 17.0
        heading = -np.pi / 4.0  # -45 degrees
        curr_vx = speed * np.cos(heading)
        curr_vy = speed * np.sin(heading)

        t_turn_start = min(60.0, duration_sec * 0.25)
        t_turn_end = min(120.0, duration_sec * 0.5)
        turn_rate = (np.pi / 4.0) / (t_turn_end - t_turn_start)  # turn 45 deg over interval

        for i, t in enumerate(time_steps):
            if i == 0:
                traj3[i] = [t, curr_x, curr_y, curr_vx, curr_vy]
                continue

            dt_step = time_steps[i] - time_steps[i - 1]
            if t_turn_start <= t <= t_turn_end:
                heading += turn_rate * dt_step
                curr_vx = speed * np.cos(heading)
                curr_vy = speed * np.sin(heading)

            curr_x += curr_vx * dt_step
            curr_y += curr_vy * dt_step
            traj3[i] = [t, curr_x, curr_y, curr_vx, curr_vy]

        vessel3 = TacticalVessel(
            track_id=3,
            vessel_type="dark_warship",
            length_m=155.0,
            has_ais=False,
            ais_spoofed=False,
            emitter_active=True,
            emitter_id="RADAR-MPQ-TYPE052D",
            mmsi=0,
            trajectory=traj3,
        )

        vessels = [vessel1, vessel2, vessel3]

        # -------------------------------------------------------------
        # Sensor Measurements Synthesis
        # -------------------------------------------------------------
        radar_measurements: List[Dict[str, Any]] = []
        esm_measurements: List[Dict[str, Any]] = []
        ais_messages: List[Dict[str, Any]] = []

        last_ais_t = -999.0

        for i, t in enumerate(time_steps):
            # 1. Radar feed (All targets reflective to radar)
            for v in vessels:
                state = v.trajectory[i]
                tx, ty, tvx, tvy = state[1], state[2], state[3], state[4]
                r_true = np.hypot(tx, ty)
                theta_true = np.arctan2(ty, tx)
                rr_true = (tx * tvx + ty * tvy) / (r_true + 1e-6)

                # Add sensor noise
                r_meas = float(r_true + self.rng.normal(0, self.radar_range_std))
                theta_meas = float(theta_true + self.rng.normal(0, self.radar_azimuth_std))
                rr_meas = float(rr_true + self.rng.normal(0, self.radar_rr_std))

                radar_measurements.append({
                    "timestamp": float(t),
                    "true_track_id": v.track_id,
                    "range_m": r_meas,
                    "azimuth_rad": theta_meas,
                    "range_rate_mps": rr_meas,
                    "meas_cartesian": np.array([r_meas * np.cos(theta_meas), r_meas * np.sin(theta_meas)]),
                })

            # 2. ESM Station (Bearing-only for active emitters)
            for v in vessels:
                if v.emitter_active:
                    state = v.trajectory[i]
                    tx, ty = state[1], state[2]
                    theta_true = np.arctan2(ty, tx)
                    esm_bearing = float(theta_true + self.rng.normal(0, self.esm_bearing_std))
                    esm_measurements.append({
                        "timestamp": float(t),
                        "true_track_id": v.track_id,
                        "bearing_rad": esm_bearing,
                        "bearing_deg": float(np.degrees(esm_bearing)),
                        "emitter_id": v.emitter_id,
                        "snr_db": float(self.rng.uniform(18.0, 28.0)),
                    })

            # 3. AIS Station (Periodic broadcast for non-dark vessels)
            if t - last_ais_t >= self.ais_period_sec - 1e-4:
                last_ais_t = t
                for v in vessels:
                    if v.has_ais:
                        state = v.trajectory[i]
                        tx, ty, tvx, tvy = state[1], state[2], state[3], state[4]
                        
                        # Apply spoofing offset if spoofed
                        if v.ais_spoofed:
                            tx_rep = tx + 3500.0
                            ty_rep = ty + 2500.0
                        else:
                            tx_rep = tx
                            ty_rep = ty

                        lat_rep, lon_rep = self.enu_to_geodetic(tx_rep, ty_rep)
                        sog_knots = float(np.hypot(tvx, tvy) * 1.94384)
                        cog_deg = float((90.0 - np.degrees(np.arctan2(tvy, tvx))) % 360.0)

                        ais_messages.append({
                            "timestamp": float(t),
                            "mmsi": v.mmsi,
                            "true_track_id": v.track_id,
                            "lat": lat_rep,
                            "lon": lon_rep,
                            "x": tx_rep,
                            "y": ty_rep,
                            "sog_knots": sog_knots,
                            "cog_deg": cog_deg,
                            "is_spoofed": v.ais_spoofed,
                        })

        sensor_data = {
            "radar_measurements": radar_measurements,
            "esm_measurements": esm_measurements,
            "ais_messages": ais_messages,
        }

        return vessels, sensor_data
