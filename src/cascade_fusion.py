import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.coord_transform import CoordinateTransformer

@dataclass
class TrajectoryState:
    x: float
    y: float
    vx: float
    vy: float
    azimuth: float  # radians
    entry_time: int
    length: int

@dataclass
class FusedTrack:
    fusion_id: int
    fusion_type: str  # 'tri', 'dual_ais_rad', 'dual_rad_vis', 'dual_ais_vis', 'single_vis'
    mmsi: Optional[int]
    radar_id: Optional[int]
    camera_id: Optional[int]
    camera_conf: float
    bbox_vis: Optional[List[float]]
    bbox_rad: Optional[List[float]]
    lat: float
    lon: float
    sog: float
    cog: float

class CascadeFusion:
    """
    Multi-Source Trajectories Fusion Module implementing Cascade Association Matching
    (Sensors 2025, 25, 2179, Section 3.2).
    """
    def __init__(self, coord_transformer: Optional[CoordinateTransformer] = None, match_cost_thresh: float = 3.0):
        self.transformer = coord_transformer or CoordinateTransformer()
        self.match_cost_thresh = match_cost_thresh
        self.fusion_counter = 0
        self.radar_mmsi_history: Dict[int, int] = {}

    def compute_cost(self, s1: TrajectoryState, s2: TrajectoryState, k: float = 0.5, l_limit: int = 100, t_limit: int = 60000) -> float:
        """
        Association cost calculation between two trajectory states (Eq. 12 & 13).
        """
        if abs(s1.length - s2.length) > l_limit or abs(s1.entry_time - s2.entry_time) > t_limit:
            return float('inf')

        # Spatial distance
        dist = math.hypot(s1.x - s2.x, s1.y - s2.y)
        
        # Velocity magnitude difference
        v1_mag = math.hypot(s1.vx, s1.vy)
        v2_mag = math.hypot(s2.vx, s2.vy)

        if v1_mag > 0.05 and v2_mag > 0.05:
            diff_v = abs(v1_mag - v2_mag)
            diff_az = abs(s1.azimuth - s2.azimuth)
            if diff_az > math.pi:
                diff_az = 2.0 * math.pi - diff_az
            diff_az_norm = diff_az / math.pi

            # Cosine distance of velocity vectors (Eq. 13)
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
        ais_records: List[Dict],
        radar_tracks: List[Dict],
        visible_tracks: List[Dict],
        ts: int
    ) -> List[FusedTrack]:
        """
        Performs 4-step cascade association matching across AIS, Radar, and Visible tracks.
        """
        fused_tracks: List[FusedTrack] = []

        # Convert all to common Radar space for unified association
        # 1. AIS states in radar space
        ais_states: List[Tuple[Dict, TrajectoryState]] = []
        for a in ais_records:
            xr, yr = self.transformer.geodetic_to_radar(a['lat'], a['lon'])
            # Extrapolate 1000ms ahead to determine velocity vector in radar coordinate space
            from src.ais_decoder import AISMessage, extrapolate_kinematic
            m_temp = AISMessage(
                mmsi=a['mmsi'],
                lat=a['lat'],
                lon=a['lon'],
                sog=a.get('sog', 0.0),
                cog=a.get('cog', 0.0),
                ts=ts
            )
            lat_next, lon_next = extrapolate_kinematic(m_temp, target_ts=ts + 1000)
            xr_next, yr_next = self.transformer.geodetic_to_radar(lat_next, lon_next)
            vx_rad = xr_next - xr
            vy_rad = yr_next - yr
            v_mag_rad = math.hypot(vx_rad, vy_rad)
            if v_mag_rad > 1e-4:
                az_rad = math.atan2(vx_rad, vy_rad)
            else:
                az_rad = 0.0

            state = TrajectoryState(
                x=xr / 2048.0,
                y=yr / 2048.0,
                vx=vx_rad / 20.0,
                vy=vy_rad / 20.0,
                azimuth=az_rad,
                entry_time=a.get('ts', ts),
                length=10
            )
            ais_states.append((a, state))

        # 2. Radar states
        radar_states: List[Tuple[Dict, TrajectoryState]] = []
        for r in radar_tracks:
            bbox = r['bbox']
            xc = bbox[0] + bbox[2] / 2.0
            yc = bbox[1] + bbox[3] / 2.0
            traj = r.get('trajectory', [])
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

            state = TrajectoryState(
                x=xc / 2048.0,
                y=yc / 2048.0,
                vx=vx,
                vy=vy,
                azimuth=az,
                entry_time=r.get('ts', ts),
                length=len(traj) if traj else 10
            )
            radar_states.append((r, state))

        # 3. Visible states mapped into radar space via inverse projection
        vis_states: List[Tuple[Dict, TrajectoryState]] = []
        for v in visible_tracks:
            bbox = v['bbox']
            xv_c = bbox[0] + bbox[2] / 2.0
            # Approximate radar horizontal mapping
            xr_est = 2300.0 - (xv_c / 2560.0) * 800.0
            yr_est = 1150.0
            state = TrajectoryState(
                x=xr_est / 2048.0,
                y=yr_est / 2048.0,
                vx=0.2,
                vy=0.2,
                azimuth=0.0,
                entry_time=v.get('ts', ts),
                length=10
            )
            vis_states.append((v, state))

        # --- STEP 1: Pairwise AIS - Radar Association (k = 0.8, prioritizing spatial accuracy) ---
        matched_ais_rad: Dict[int, int] = {}  # ais_idx -> rad_idx
        unmatched_ais = list(range(len(ais_states)))
        unmatched_rad = list(range(len(radar_states)))

        if len(ais_states) > 0 and len(radar_states) > 0:
            cost_ar = np.zeros((len(ais_states), len(radar_states)), dtype=np.float32)
            for i, (a_data, s_a) in enumerate(ais_states):
                for j, (r_data, s_r) in enumerate(radar_states):
                    c = self.compute_cost(s_a, s_r, k=0.8)
                    dist_px = math.hypot(s_a.x - s_r.x, s_a.y - s_r.y) * 2048.0
                    r_tid = r_data.get('track_id')
                    if r_tid in self.radar_mmsi_history and self.radar_mmsi_history[r_tid] == a_data['mmsi'] and dist_px < 50.0:
                        c -= 0.5  # Temporal continuity hysteresis
                    cost_ar[i, j] = c

            r_ind, c_ind = linear_sum_assignment(cost_ar)
            for r, c in zip(r_ind, c_ind):
                if cost_ar[r, c] < self.match_cost_thresh:
                    matched_ais_rad[r] = c
                    r_tid = radar_states[c][0].get('track_id')
                    a_mmsi = ais_states[r][0]['mmsi']
                    self.radar_mmsi_history[r_tid] = a_mmsi
                    if r in unmatched_ais:
                        unmatched_ais.remove(r)
                    if c in unmatched_rad:
                        unmatched_rad.remove(c)

        # --- STEP 2: Pairwise Radar - Visible Association (k = 0.2) ---
        matched_rad_vis: Dict[int, int] = {}  # rad_idx -> vis_idx
        unmatched_vis = list(range(len(vis_states)))

        if len(radar_states) > 0 and len(vis_states) > 0:
            cost_rv = np.zeros((len(radar_states), len(vis_states)), dtype=np.float32)
            for i, (r_data, s_r) in enumerate(radar_states):
                # Project radar bbox to visible gate
                v_gate = self.transformer.radar_bbox_to_visible_gate(r_data['bbox'])
                for j, (v_data, _) in enumerate(vis_states):
                    v_box = v_data['bbox']
                    # Calculate gate distance
                    dist_x = abs((v_box[0] + v_box[2]/2.0) - (v_gate[0] + v_gate[2]/2.0)) / 2560.0
                    dist_y = abs((v_box[1] + v_box[3]/2.0) - (v_gate[1] + v_gate[2]/2.0)) / 1440.0
                    cost_rv[i, j] = 0.2 * dist_x + 0.8 * dist_y

            r_ind, c_ind = linear_sum_assignment(cost_rv)
            for r, c in zip(r_ind, c_ind):
                if cost_rv[r, c] < 0.6:  # Projection gate threshold
                    matched_rad_vis[r] = c
                    if c in unmatched_vis:
                        unmatched_vis.remove(c)

        # --- STEP 3: Unassociated AIS (TU_AA) - Unassociated Visible (TU_AV) ---
        matched_ais_vis: Dict[int, int] = {}
        for a_idx in list(unmatched_ais):
            a_data, _ = ais_states[a_idx]
            xv_proj, yv_proj = self.transformer.geodetic_to_visible(a_data['lat'], a_data['lon'])
            for v_idx in list(unmatched_vis):
                v_box = vis_states[v_idx][0]['bbox']
                if abs(xv_proj - (v_box[0] + v_box[2]/2.0)) < 300.0:  # Proximity match
                    matched_ais_vis[a_idx] = v_idx
                    unmatched_ais.remove(a_idx)
                    unmatched_vis.remove(v_idx)
                    break

        # --- STEP 4: Integrated Matching & Output Synthesis ---
        # 4a. Tri-fusion (AIS + Radar + Visible)
        assigned_rad = set()
        assigned_vis = set()
        assigned_ais = set()

        for a_idx, r_idx in matched_ais_rad.items():
            if r_idx in matched_rad_vis:
                v_idx = matched_rad_vis[r_idx]
                a_data = ais_states[a_idx][0]
                r_data = radar_states[r_idx][0]
                v_data = vis_states[v_idx][0]
                
                self.fusion_counter += 1
                fused_tracks.append(FusedTrack(
                    fusion_id=self.fusion_counter,
                    fusion_type='tri',
                    mmsi=a_data['mmsi'],
                    radar_id=r_data['track_id'],
                    camera_id=v_data['track_id'],
                    camera_conf=v_data.get('conf', 1.0),
                    bbox_vis=v_data['bbox'],
                    bbox_rad=r_data['bbox'],
                    lat=a_data['lat'],
                    lon=a_data['lon'],
                    sog=a_data.get('sog', 0.0),
                    cog=a_data.get('cog', 0.0)
                ))
                assigned_ais.add(a_idx)
                assigned_rad.add(r_idx)
                assigned_vis.add(v_idx)

        # 4b. Dual: Radar + Visible (no AIS)
        for r_idx, v_idx in matched_rad_vis.items():
            if r_idx not in assigned_rad and v_idx not in assigned_vis:
                r_data = radar_states[r_idx][0]
                v_data = vis_states[v_idx][0]
                self.fusion_counter += 1
                fused_tracks.append(FusedTrack(
                    fusion_id=self.fusion_counter,
                    fusion_type='dual_rad_vis',
                    mmsi=None,
                    radar_id=r_data['track_id'],
                    camera_id=v_data['track_id'],
                    camera_conf=v_data.get('conf', 1.0),
                    bbox_vis=v_data['bbox'],
                    bbox_rad=r_data['bbox'],
                    lat=0.0,
                    lon=0.0,
                    sog=0.0,
                    cog=0.0
                ))
                assigned_rad.add(r_idx)
                assigned_vis.add(v_idx)

        # 4c. Dual: AIS + Visible (no Radar)
        for a_idx, v_idx in matched_ais_vis.items():
            if a_idx not in assigned_ais and v_idx not in assigned_vis:
                a_data = ais_states[a_idx][0]
                v_data = vis_states[v_idx][0]
                self.fusion_counter += 1
                fused_tracks.append(FusedTrack(
                    fusion_id=self.fusion_counter,
                    fusion_type='dual_ais_vis',
                    mmsi=a_data['mmsi'],
                    radar_id=None,
                    camera_id=v_data['track_id'],
                    camera_conf=v_data.get('conf', 1.0),
                    bbox_vis=v_data['bbox'],
                    bbox_rad=None,
                    lat=a_data['lat'],
                    lon=a_data['lon'],
                    sog=a_data.get('sog', 0.0),
                    cog=a_data.get('cog', 0.0)
                ))
                assigned_ais.add(a_idx)
                assigned_vis.add(v_idx)

        # 4d. Single: Unassociated Visible tracks
        for v_idx in unmatched_vis:
            if v_idx not in assigned_vis:
                v_data = vis_states[v_idx][0]
                self.fusion_counter += 1
                fused_tracks.append(FusedTrack(
                    fusion_id=self.fusion_counter,
                    fusion_type='single_vis',
                    mmsi=None,
                    radar_id=None,
                    camera_id=v_data['track_id'],
                    camera_conf=v_data.get('conf', 1.0),
                    bbox_vis=v_data['bbox'],
                    bbox_rad=None,
                    lat=0.0,
                    lon=0.0,
                    sog=0.0,
                    cog=0.0
                ))

        return fused_tracks
