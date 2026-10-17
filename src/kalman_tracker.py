import numpy as np
from typing import List, Optional

def convert_bbox_to_z(bbox: List[float]) -> np.ndarray:
    """
    Takes a bounding box in the form [x, y, w, h] and returns z in the form
    [x_center, y_center, aspect_ratio, h] where aspect_ratio = w / h.
    """
    w = max(bbox[2], 1.0)
    h = max(bbox[3], 1.0)
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w / h  # aspect ratio
    return np.array([x, y, s, h]).reshape((4, 1))

def convert_x_to_bbox(x: np.ndarray) -> List[float]:
    """
    Takes a state vector [x_center, y_center, s, h, ...] and returns bounding box
    [x_min, y_min, w, h].
    """
    xc = float(x[0, 0])
    yc = float(x[1, 0])
    s = float(x[2, 0])
    h = float(x[3, 0])
    w = max(s * h, 1.0)
    h = max(h, 1.0)
    xmin = xc - w / 2.0
    ymin = yc - h / 2.0
    return [xmin, ymin, w, h]

class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes in image space.
    State: [xc, yc, s, h, vx, vy, vs, vh]
    """
    count = 0

    def __init__(self, bbox: List[float], track_id: Optional[int] = None):
        if track_id is None:
            KalmanBoxTracker.count += 1
            self.track_id = KalmanBoxTracker.count
        else:
            self.track_id = track_id

        # 8 state dimensions, 4 measurement dimensions
        self.dim_x = 8
        self.dim_z = 4

        # State vector
        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = convert_bbox_to_z(bbox)

        # State transition matrix F
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty in initial velocities

        self.Q = np.eye(self.dim_x) * 1.0
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(self.dim_z) * 1.0

        self.time_since_update = 0
        self.history = [bbox]
        self.hits = 1
        self.age = 1
        self.confidence_history = [1.0]

    def predict(self) -> List[float]:
        """Advances state vector and returns predicted bounding box."""
        # Ensure aspect ratio and height remain positive
        if self.x[3] + self.x[7] <= 0:
            self.x[7] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        pred_bbox = convert_x_to_bbox(self.x)
        return pred_bbox

    def update(self, bbox: List[float], conf: float = 1.0):
        """Updates the state vector with observed bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.confidence_history.append(conf)
        z = convert_bbox_to_z(bbox)
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
        
        cur_bbox = convert_x_to_bbox(self.x)
        self.history.append(cur_bbox)

    def get_state(self) -> List[float]:
        """Returns the current bounding box estimate."""
        return convert_x_to_bbox(self.x)
