"""2D constant-velocity Kalman filter with Rauch–Tung–Striebel smoothing."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from . import config as cfg


def _process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """Discrete white-noise acceleration process covariance for 2D CV."""
    dt = float(max(dt, 1e-6))
    q = sigma_a**2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    q11 = 0.25 * dt4 * q
    q13 = 0.5 * dt3 * q
    q33 = dt2 * q
    Q = np.array(
        [
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33],
        ],
        dtype=np.float64,
    )
    return Q


def _transition(dt: float) -> np.ndarray:
    dt = float(max(dt, 1e-6))
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def kalman_rts_smooth_2d(
    x: np.ndarray,
    y: np.ndarray,
    t_s: np.ndarray,
    sigma_gps: float = cfg.GPS_SIGMA_M,
    sigma_a: float = cfg.PROCESS_ACCEL_SIGMA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward Kalman filter + RTS smoother for 2D constant-velocity dynamics.

    Returns
    -------
    x_hat, y_hat, vx_hat, vy_hat, speed_hat
    """
    n = len(x)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    R = (sigma_gps**2) * np.eye(2, dtype=np.float64)
    I4 = np.eye(4, dtype=np.float64)

    # Storage
    x_filt = np.zeros((n, 4), dtype=np.float64)
    P_filt = np.zeros((n, 4, 4), dtype=np.float64)
    x_pred = np.zeros((n, 4), dtype=np.float64)
    P_pred = np.zeros((n, 4, 4), dtype=np.float64)
    F_store = np.zeros((n, 4, 4), dtype=np.float64)

    # Initialize from first measurement; velocity from first step if available
    z0 = np.array([x[0], y[0]], dtype=np.float64)
    if n >= 2:
        dt0 = max(float(t_s[1] - t_s[0]), 1e-6)
        v0 = np.array([(x[1] - x[0]) / dt0, (y[1] - y[0]) / dt0], dtype=np.float64)
    else:
        v0 = np.zeros(2, dtype=np.float64)

    x_filt[0] = np.array([z0[0], z0[1], v0[0], v0[1]], dtype=np.float64)
    P_filt[0] = np.diag([sigma_gps**2, sigma_gps**2, 25.0, 25.0]).astype(np.float64)
    x_pred[0] = x_filt[0].copy()
    P_pred[0] = P_filt[0].copy()
    F_store[0] = I4

    # Forward pass
    for k in range(1, n):
        dt = float(t_s[k] - t_s[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 1e-6
        F = _transition(dt)
        Q = _process_noise(dt, sigma_a)
        F_store[k] = F

        x_pred[k] = F @ x_filt[k - 1]
        P_pred[k] = F @ P_filt[k - 1] @ F.T + Q

        z = np.array([x[k], y[k]], dtype=np.float64)
        innov = z - H @ x_pred[k]
        S = H @ P_pred[k] @ H.T + R
        # Solve K = P H^T S^{-1} stably
        K = np.linalg.solve(S, (P_pred[k] @ H.T).T).T
        x_filt[k] = x_pred[k] + K @ innov
        P_filt[k] = (I4 - K @ H) @ P_pred[k]

    # RTS backward pass
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for k in range(n - 2, -1, -1):
        F_next = F_store[k + 1]
        # C = P_filt[k] F^T P_pred[k+1]^{-1}
        C = np.linalg.solve(P_pred[k + 1].T, (P_filt[k] @ F_next.T).T).T
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T

    x_hat = x_smooth[:, 0]
    y_hat = x_smooth[:, 1]
    vx_hat = x_smooth[:, 2]
    vy_hat = x_smooth[:, 3]
    speed_hat = np.sqrt(vx_hat**2 + vy_hat**2)
    return x_hat, y_hat, vx_hat, vy_hat, speed_hat


def compute_tangential_accel(
    speed: np.ndarray,
    delta_t: np.ndarray,
    clip_min: float = cfg.ACCEL_CLIP_MIN,
    clip_max: float = cfg.ACCEL_CLIP_MAX,
) -> np.ndarray:
    """Smoothed tangential acceleration from successive speeds, clipped to physical bounds."""
    a = np.zeros_like(speed, dtype=np.float64)
    if len(speed) < 2:
        return a
    dt = delta_t[1:].astype(np.float64)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, np.nan)
    raw = (speed[1:] - speed[:-1]) / dt
    raw = np.where(np.isfinite(raw), raw, 0.0)
    a[1:] = np.clip(raw, clip_min, clip_max)
    a[0] = a[1] if len(a) > 1 else 0.0
    return a


def smooth_segment(seg: pd.DataFrame) -> pd.DataFrame:
    """Apply Kalman+RTS smoothing and attach kinematics to a segment DataFrame."""
    out = seg.copy()
    x_hat, y_hat, vx_hat, vy_hat, speed_hat = kalman_rts_smooth_2d(
        out["x"].to_numpy(dtype=np.float64),
        out["y"].to_numpy(dtype=np.float64),
        out["t_s"].to_numpy(dtype=np.float64),
    )
    out["x_hat"] = x_hat
    out["y_hat"] = y_hat
    out["vx_hat"] = vx_hat
    out["vy_hat"] = vy_hat
    out["smoothed_speed"] = speed_hat
    out["smoothed_accel"] = compute_tangential_accel(
        speed_hat, out["delta_t"].to_numpy(dtype=np.float64)
    )
    return out


def smooth_segments(segments: List[dict]) -> List[pd.DataFrame]:
    """Smooth a list of segment records; returns list of DataFrames."""
    from .preprocessing import record_to_dataframe

    smoothed: List[pd.DataFrame] = []
    for rec in segments:
        seg = record_to_dataframe(rec) if isinstance(rec, dict) else rec
        smoothed.append(smooth_segment(seg))
    return smoothed


def smooth_segment_record(rec: dict) -> dict:
    """Smooth one segment record in-place-style, returning an enriched dict."""
    from .preprocessing import record_to_dataframe

    seg = smooth_segment(record_to_dataframe(rec))
    return {
        **rec,
        "x_hat": seg["x_hat"].to_numpy(dtype=np.float64),
        "y_hat": seg["y_hat"].to_numpy(dtype=np.float64),
        "smoothed_speed": seg["smoothed_speed"].to_numpy(dtype=np.float64),
        "smoothed_accel": seg["smoothed_accel"].to_numpy(dtype=np.float64),
    }
