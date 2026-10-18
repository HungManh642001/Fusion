"""Decoupled Unscented Kalman Filter Core for Marine Tracking.

Implements a 9-sigma-point Unscented Transform with Aidala-Hammel velocity decoupling
(K_vel = 0) for bearing-only updates, preventing spurious velocity divergence in maritime environments.
"""

from typing import Tuple
import numpy as np


class DecoupledUKF:
    """Decoupled Unscented Kalman Filter for 2D marine kinematic state tracking."""

    def __init__(
        self,
        init_pos: np.ndarray,
        is_large_ship: bool = False,
        q: float = 0.04,
        r_pos: float = 16.0,
        max_speed: float = 12.0,
    ) -> None:
        """Initialize the tracker state and covariances.

        Args:
            init_pos: Initial [North, East] coordinates (meters).
            is_large_ship: Flag indicating large vessel (higher inertia, lower process noise).
            q: Acceleration spectral noise density (m^2 / s^3).
            r_pos: Position measurement covariance scalar (m^2).
            max_speed: Maximum allowable vessel speed (m/s).
        """
        self.is_large_ship = is_large_ship
        self.q = 0.02 if is_large_ship else q
        self.r_pos = 49.0 if is_large_ship else r_pos
        self.max_speed = max_speed

        # State: [p_N, v_N, p_E, v_E]^T
        self.x = np.array([float(init_pos[0]), 0.0, float(init_pos[1]), 0.0], dtype=float)
        self.P = np.diag([25.0, 4.0, 25.0, 4.0]).astype(float)
        self.time_since_pos = 0.0

        # UT scaling parameters
        self.n = 4
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

        # Weights
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1.0 - self.alpha**2 + self.beta)

    @property
    def position(self) -> np.ndarray:
        """Return 2D position [North, East] in meters."""
        return self.x[[0, 2]].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Return 2D velocity [v_N, v_E] in m/s."""
        return self.x[[1, 3]].copy()

    @property
    def speed(self) -> float:
        """Return vessel speed magnitude in m/s."""
        return float(np.hypot(self.x[1], self.x[3]))

    def predict(self, dt: float) -> None:
        """Propagate state forward by time interval dt using Constant Velocity model."""
        if dt <= 1e-5:
            return
        dt = min(dt, 1.0)

        F = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        q_sub = np.array([
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt],
        ], dtype=float) * self.q

        Q = np.zeros((4, 4), dtype=float)
        Q[0:2, 0:2] = q_sub
        Q[2:4, 2:4] = q_sub

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.time_since_pos += dt

        # Enforce physical speed clamp
        spd = self.speed
        if spd > self.max_speed:
            scale = self.max_speed / spd
            self.x[1] *= scale
            self.x[3] *= scale

    def update_pos(self, z_pos: np.ndarray) -> None:
        """Update filter state using 2D position measurement [North, East]."""
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
        R = np.eye(2, dtype=float) * self.r_pos
        y = z_pos - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.time_since_pos = 0.0

    def update_bearing(self, z_brg: float, own_pos: np.ndarray, gate_rad: float = np.radians(5.0)) -> bool:
        """Update cross-LOS position using 1D bearing measurement with Aidala-Hammel decoupling.

        Args:
            z_brg: Measured bearing in radians relative to North.
            own_pos: Current ownship position [North, East].
            gate_rad: Innovation gating threshold in radians.

        Returns:
            bool: True if bearing passed gating and was updated, False otherwise.
        """
        dN = self.x[0] - own_pos[0]
        dE = self.x[2] - own_pos[1]
        r2 = dN**2 + dE**2
        if r2 < 1.0:
            return False  # Zero-range singularity protection

        # Generate Sigma Points
        P_sqrt = np.linalg.cholesky((self.n + self.lam) * self.P + 1e-8 * np.eye(self.n))
        sigmas = np.zeros((2 * self.n + 1, self.n), dtype=float)
        sigmas[0] = self.x
        for i in range(self.n):
            sigmas[i + 1] = self.x + P_sqrt[:, i]
            sigmas[self.n + i + 1] = self.x - P_sqrt[:, i]

        # Transform sigma points through nonlinear bearing measurement function
        z_sigmas = np.zeros(2 * self.n + 1, dtype=float)
        for i, s in enumerate(sigmas):
            sn = s[0] - own_pos[0]
            se = s[2] - own_pos[1]
            z_sigmas[i] = np.arctan2(se, sn)

        # Predicted measurement mean (circular mean)
        sin_sum = np.sum(self.Wm * np.sin(z_sigmas))
        cos_sum = np.sum(self.Wm * np.cos(z_sigmas))
        z_pred = np.arctan2(sin_sum, cos_sum)

        # Innovation and gating
        y = (z_brg - z_pred + np.pi) % (2.0 * np.pi) - np.pi
        if abs(y) > gate_rad:
            return False

        # Innovation covariance and cross-covariance
        r_dist = np.sqrt(r2)
        r_sensor_noise = (0.035 * max(r_dist / 100.0, 0.5)) ** 2
        S = r_sensor_noise
        P_xz = np.zeros(self.n, dtype=float)

        for i in range(2 * self.n + 1):
            dz = (z_sigmas[i] - z_pred + np.pi) % (2.0 * np.pi) - np.pi
            dx = sigmas[i] - self.x
            S += self.Wc[i] * dz**2
            P_xz += self.Wc[i] * dx * dz

        # Kalman gain computation
        K = P_xz / max(S, 1e-6)

        # Aidala-Hammel velocity decoupling: strictly zero out velocity gain!
        K[1] = 0.0
        K[3] = 0.0

        # Update state and covariance
        self.x += K * y
        self.P -= np.outer(K, K) * S
        return True
