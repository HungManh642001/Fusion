"""Interacting Multiple Model (IMM) Kalman Filter for Maneuvering Naval Targets.

Combines:
1. Constant Velocity (CV) model for straight-line merchant transit.
2. Coordinated Turn (CT) model for evasive/tactical warship maneuvers.

Uses Markovian probability mixing to dynamically adapt to unexpected tactical turns.
"""

from typing import Tuple, List, Optional
import numpy as np


class IMMTracker:
    """IMM filter coordinating CV and CT kinematic models."""

    def __init__(
        self,
        init_pos: np.ndarray,
        init_vel: np.ndarray = np.array([0.0, 0.0]),
        dt: float = 2.0,
        turn_rate_rad_s: float = 0.02,
        r_meas_noise: float = 15.0,
    ):
        self.dt = dt
        self.omega = turn_rate_rad_s
        self.M = 2  # 2 models: 0 -> CV, 1 -> CT

        # Initial combined state: [x, vx, y, vy]
        self.x = np.array([init_pos[0], init_vel[0], init_pos[1], init_vel[1]], dtype=float)
        self.P = np.diag([r_meas_noise**2, 25.0, r_meas_noise**2, 25.0]).astype(float)

        # Per-model states and covariances
        self.x_models = [self.x.copy(), self.x.copy()]
        self.P_models = [self.P.copy(), self.P.copy()]

        # Mode probabilities: mu[0] = P(CV), mu[1] = P(CT)
        self.mu = np.array([0.80, 0.20], dtype=float)

        # Mode transition probability matrix Pi
        self.Pi = np.array([
            [0.95, 0.05],  # CV -> CV (0.95), CV -> CT (0.05)
            [0.10, 0.90],  # CT -> CV (0.10), CT -> CT (0.90)
        ], dtype=float)

        # Transition matrices
        self.F_cv = self._build_cv_matrix(dt)
        self.F_ct = self._build_ct_matrix(dt, self.omega)

        # Process noise covariance matrices Q
        q_cv = 0.2  # m^2/s^3
        q_ct = 2.0  # m^2/s^3 (higher uncertainty for high-G maneuvers)
        self.Q_cv = self._build_q_matrix(dt, q_cv)
        self.Q_ct = self._build_q_matrix(dt, q_ct)

        # Measurement matrix H for 2D Cartesian radar [x, y]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=float)

        self.R_radar = np.diag([r_meas_noise**2, r_meas_noise**2]).astype(float)

    @staticmethod
    def _build_cv_matrix(dt: float) -> np.ndarray:
        return np.array([
            [1.0,  dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0,  dt],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _build_ct_matrix(dt: float, omega: float) -> np.ndarray:
        if abs(omega) < 1e-5:
            return IMMTracker._build_cv_matrix(dt)
        s = np.sin(omega * dt)
        c = np.cos(omega * dt)
        return np.array([
            [1.0,  s / omega,       0.0, -(1.0 - c) / omega],
            [0.0,          c,       0.0,                 -s],
            [0.0, (1.0 - c) / omega, 1.0,          s / omega],
            [0.0,          s,       0.0,                  c],
        ], dtype=float)

    @staticmethod
    def _build_q_matrix(dt: float, q: float) -> np.ndarray:
        dt2 = dt**2 / 2.0
        dt3 = dt**3 / 3.0
        dt4 = dt**4 / 4.0
        Q1d = np.array([[dt4, dt3], [dt3, dt**2]], dtype=float) * q
        Q = np.zeros((4, 4), dtype=float)
        Q[0:2, 0:2] = Q1d
        Q[2:4, 2:4] = Q1d
        return Q

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform IMM interaction mixing followed by model predictions."""
        # 1. Interaction / Mixing
        c_bar = self.Pi.T @ self.mu  # shape: (2,)
        # Avoid zero division
        c_bar = np.maximum(c_bar, 1e-12)

        mu_mix = np.zeros((self.M, self.M), dtype=float)
        for i in range(self.M):
            for j in range(self.M):
                mu_mix[i, j] = (self.Pi[i, j] * self.mu[i]) / c_bar[j]

        # Mixed initial condition for each model j
        x_0 = [np.zeros(4, dtype=float) for _ in range(self.M)]
        P_0 = [np.zeros((4, 4), dtype=float) for _ in range(self.M)]

        for j in range(self.M):
            for i in range(self.M):
                x_0[j] += mu_mix[i, j] * self.x_models[i]

            for i in range(self.M):
                dx = (self.x_models[i] - x_0[j]).reshape(-1, 1)
                P_0[j] += mu_mix[i, j] * (self.P_models[i] + dx @ dx.T)

        # 2. Model-Conditioned Prediction
        F_list = [self.F_cv, self.F_ct]
        Q_list = [self.Q_cv, self.Q_ct]

        for j in range(self.M):
            self.x_models[j] = F_list[j] @ x_0[j]
            self.P_models[j] = F_list[j] @ P_0[j] @ F_list[j].T + Q_list[j]

        # 3. Overall Combined Prediction
        self.x = np.zeros(4, dtype=float)
        self.P = np.zeros((4, 4), dtype=float)
        for j in range(self.M):
            self.x += self.mu[j] * self.x_models[j]

        for j in range(self.M):
            dx = (self.x_models[j] - self.x).reshape(-1, 1)
            self.P += self.mu[j] * (self.P_models[j] + dx @ dx.T)

        return self.x.copy(), self.P.copy()

    def update_radar(self, z_cartesian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update tracker with 2D Cartesian radar measurement [x, y]."""
        z = np.asarray(z_cartesian, dtype=float)
        likelihoods = np.zeros(self.M, dtype=float)

        for j in range(self.M):
            x_j = self.x_models[j]
            P_j = self.P_models[j]

            # Innovation
            y_j = z - self.H @ x_j
            S_j = self.H @ P_j @ self.H.T + self.R_radar

            # Kalman gain
            inv_S = np.linalg.inv(S_j)
            K_j = P_j @ self.H.T @ inv_S

            # Updated state and covariance
            self.x_models[j] = x_j + K_j @ y_j
            I_KH = np.eye(4) - K_j @ self.H
            self.P_models[j] = I_KH @ P_j @ I_KH.T + K_j @ self.R_radar @ K_j.T

            # Gaussian Likelihood Lambda_j
            det_S = np.linalg.det(S_j)
            d2 = float(y_j.T @ inv_S @ y_j)
            denom = 2.0 * np.pi * np.sqrt(max(det_S, 1e-12))
            likelihoods[j] = np.exp(-0.5 * min(d2, 50.0)) / max(denom, 1e-12)

        # 3. Model Probability Update
        c_bar = self.Pi.T @ self.mu
        numerators = c_bar * likelihoods
        total_sum = np.sum(numerators)
        if total_sum > 1e-15:
            self.mu = numerators / total_sum
        else:
            self.mu = np.array([0.5, 0.5])

        # 4. Overall Combined State Update
        self.x = np.zeros(4, dtype=float)
        self.P = np.zeros((4, 4), dtype=float)
        for j in range(self.M):
            self.x += self.mu[j] * self.x_models[j]

        for j in range(self.M):
            dx = (self.x_models[j] - self.x).reshape(-1, 1)
            self.P += self.mu[j] * (self.P_models[j] + dx @ dx.T)

        return self.x.copy(), self.P.copy()
