import math
from typing import List, Tuple
import numpy as np

# Calibrated Affine Matrix from Geodetic [lon, lat, 1] to Radar [xr, yr]
# Minimizes reprojection error to 1.14 px on 2048x2048 radar image
AFFINE_GEO_TO_RADAR = np.array([
    [-5987.59411024, -13375.50157462, 1094999.35764672],
    [-1240.64398706,  -4834.35908282,  290753.26076505]
], dtype=np.float64)

# Calibrated Projection Matrix F from Radar [xr, yr, 1] to Visible Bbox [xv, yv, wv, hv]
# on 2560x1440 camera resolution
F_RADAR_TO_VISIBLE = np.array([
    [-1.38157877e+01, -8.78508291e+00,  3.51290525e+04],
    [ 2.31098562e-01, -2.83131713e-01,  6.87422028e+02],
    [ 2.00502139e+00,  1.84434496e+00, -5.27668883e+03],
    [ 2.13125082e-01,  2.51459832e-01, -6.02300267e+02]
], dtype=np.float64)

class CoordinateTransformer:
    """Handles spatiotemporal coordinate transformations between Geodetic (Lat/Lon), Radar PPI, and Visible Camera."""
    def __init__(self, platform_lat: float = 30.563549, platform_lon: float = 114.306087, platform_h: float = 20.0):
        self.platform_lat = platform_lat
        self.platform_lon = platform_lon
        self.platform_h = platform_h
        self.M_geo2rad = AFFINE_GEO_TO_RADAR
        self.F_rad2vis = F_RADAR_TO_VISIBLE

    def geodetic_to_radar(self, lat: float, lon: float) -> Tuple[float, float]:
        """Maps geodetic (lat, lon) coordinates to radar pixel coordinates (xr, yr)."""
        pt = np.array([lon, lat, 1.0], dtype=np.float64)
        res = self.M_geo2rad @ pt
        return float(res[0]), float(res[1])

    def radar_to_visible(self, xr: float, yr: float) -> Tuple[float, float]:
        """Maps radar pixel coordinates (xr, yr) to visible camera center (xv, yv)."""
        pt = np.array([xr, yr, 1.0], dtype=np.float64)
        bbox = self.F_rad2vis @ pt
        xv_center = bbox[0] + bbox[2] / 2.0
        yv_center = bbox[1] + bbox[3] / 2.0
        return float(xv_center), float(yv_center)

    def radar_bbox_to_visible_gate(self, bbox_rad: List[float], expansion_factor: float = 1.2) -> List[float]:
        """
        Projects a radar bounding box [xmin, ymin, w, h] to visible image tracking gate [xmin, ymin, w, h].
        Expands the gate slightly by expansion_factor to accommodate projection uncertainty.
        """
        xr_c = bbox_rad[0] + bbox_rad[2] / 2.0
        yr_c = bbox_rad[1] + bbox_rad[3] / 2.0
        pt = np.array([xr_c, yr_c, 1.0], dtype=np.float64)
        v_box = self.F_rad2vis @ pt
        
        # Apply expansion factor for tracking gate
        x, y, w, h = v_box[0], v_box[1], v_box[2], v_box[3]
        w_exp = max(w * expansion_factor, 60.0)
        h_exp = max(h * expansion_factor, 30.0)
        x_exp = x - (w_exp - w) / 2.0
        y_exp = y - (h_exp - h) / 2.0
        
        # Clip to image boundary (2560 x 1440)
        x_exp = max(0.0, min(2550.0, x_exp))
        y_exp = max(0.0, min(1430.0, y_exp))
        return [float(x_exp), float(y_exp), float(w_exp), float(h_exp)]

    def geodetic_to_visible(self, lat: float, lon: float) -> Tuple[float, float]:
        """Direct projection from geodetic to visible coordinates via intermediate radar projection."""
        xr, yr = self.geodetic_to_radar(lat, lon)
        return self.radar_to_visible(xr, yr)

    def is_in_msf_fov(self, lat: float, lon: float) -> bool:
        """
        Determines whether a vessel target is within the camera's field of view (MSF area).
        Camera has a 54.3 deg horizontal FOV directed across the Yangtze channel.
        """
        xr, yr = self.geodetic_to_radar(lat, lon)
        xv, _ = self.radar_to_visible(xr, yr)
        # Visible image horizontal extent: 0 to 2560 px
        return -100 <= xv <= 2660
