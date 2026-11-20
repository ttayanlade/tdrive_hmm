"""Feature engineering, constrained 3-state Gaussian HMM, and Viterbi decoding."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def local_jitter_radius(
    x: np.ndarray,
    y: np.ndarray,
    t_s: np.ndarray,
    window_s: float = cfg.JITTER_WINDOW_S,
) -> np.ndarray:
    """
    90th-percentile Euclidean distance from the local time-window centroid.

    Window of width ``window_s`` is centered on each timestamp ``t``.
    """
    n = len(x)
    R = np.zeros(n, dtype=np.float64)
    if n == 0:
        return R

    half = window_s / 2.0
    left = 0
    right = 0
    for i in range(n):
        t = t_s[i]
        while left < n and t_s[left] < t - half:
            left += 1
        while right < n and t_s[right] <= t + half:
            right += 1
        # window is [left, right)
        if right - left <= 1:
            R[i] = 0.0
            continue
        xs = x[left:right]
        ys = y[left:right]
        cx = float(xs.mean())
        cy = float(ys.mean())
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        R[i] = float(np.percentile(dist, 90))
    return R


def build_observation_matrix(seg: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build emission features ``O_t = [log(v+eps), a, log(R+eps)]``.

    Returns ``(O, jitter_radius)`` where ``O`` has shape ``(T, 3)``.
    """
    x = seg["x_hat"].to_numpy(dtype=np.float64)
    y = seg["y_hat"].to_numpy(dtype=np.float64)
    t_s = seg["t_s"].to_numpy(dtype=np.float64)
    speed = np.maximum(seg["smoothed_speed"].to_numpy(dtype=np.float64), 0.0)
    accel = seg["smoothed_accel"].to_numpy(dtype=np.float64)
    R = local_jitter_radius(x, y, t_s)

    O = np.column_stack(
        [
            np.log(speed + cfg.SPEED_LOG_EPS),
            accel,
            np.log(R + cfg.JITTER_LOG_EPS),
        ]
    ).astype(np.float64)
    O = np.nan_to_num(O, nan=0.0, posinf=0.0, neginf=0.0)
    return O, R


def enrich_segment_features(seg: pd.DataFrame) -> pd.DataFrame:
    """Attach observation features and jitter radius to a smoothed segment."""
    out = seg.copy()
    O, R = build_observation_matrix(out)
    out["jitter_radius"] = R
    out["feat_log_speed"] = O[:, 0]
    out["feat_accel"] = O[:, 1]
    out["feat_log_jitter"] = O[:, 2]
    return out


def fit_scaler(train_segments: Sequence[pd.DataFrame]) -> StandardScaler:
    """Fit ``StandardScaler`` on concatenated training observations only."""
    mats = []
    for seg in train_segments:
        if "feat_log_speed" not in seg.columns:
            seg = enrich_segment_features(seg)
        mats.append(
            np.column_stack(
                [
                    seg["feat_log_speed"].to_numpy(),
                    seg["feat_accel"].to_numpy(),
                    seg["feat_log_jitter"].to_numpy(),
                ]
            )
        )
    if not mats:
        raise ValueError("No training segments available to fit StandardScaler.")
    X = np.vstack(mats)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def scale_segment(seg: pd.DataFrame, scaler: StandardScaler) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply a fitted scaler; returns (enriched_seg, scaled_O)."""
    out = seg.copy()
    if "feat_log_speed" not in out.columns or "jitter_radius" not in out.columns:
        out = enrich_segment_features(out)

    O = np.column_stack(
        [
            out["feat_log_speed"].to_numpy(),
            out["feat_accel"].to_numpy(),
            out["feat_log_jitter"].to_numpy(),
        ]
    )
    O_scaled = scaler.transform(O)
    out["o0"] = O_scaled[:, 0]
    out["o1"] = O_scaled[:, 1]
    out["o2"] = O_scaled[:, 2]
    return out, O_scaled


def _init_transition_matrix() -> np.ndarray:
    """
    Constrained topology:
      - self-loops allowed
      - stop <-> driving allowed
      - A_12 = A_21 = 0 (no direct stop-to-stop)
    Persistence priors: A_11≈0.90 (short traffic), A_22≈0.99 (long voluntary).
    """
    A = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=np.float64)
    # Driving: mostly persist, occasional entry to either stop
    A[0, 0] = 0.90
    A[0, 1] = 0.05
    A[0, 2] = 0.05
    # Involuntary stop: shorter dwell
    A[1, 1] = 0.90
    A[1, 0] = 0.10
    A[1, 2] = 0.0
    # Voluntary stop: long dwell
    A[2, 2] = 0.99
    A[2, 0] = 0.01
    A[2, 1] = 0.0
    # Row-normalize defensively
    A = A / A.sum(axis=1, keepdims=True)
    return A


def _init_means_covars(
    X: np.ndarray,
    lengths: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize emission means/covars from data statistics.

    State 0: high speed / jitter (upper quantile of features).
    States 1 & 2: near-zero speed/jitter, perturbed by ±0.05 feature stds.
    """
    # Feature columns are already standardized → mean~0, std~1
    feat_std = X.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    # Speed / jitter are features 0 and 2
    high_speed = np.quantile(X[:, 0], 0.75)
    high_jitter = np.quantile(X[:, 2], 0.75)
    low_speed = np.quantile(X[:, 0], 0.10)
    low_jitter = np.quantile(X[:, 2], 0.10)
    accel_mean = float(np.mean(X[:, 1]))

    means = np.zeros((cfg.N_STATES, 3), dtype=np.float64)
    means[0] = np.array([high_speed, accel_mean, high_jitter])
    # Near-zero speed/jitter for stop states with ±0.05 std perturbation
    means[1] = np.array(
        [
            low_speed - 0.05 * feat_std[0],
            accel_mean,
            low_jitter - 0.05 * feat_std[2],
        ]
    )
    means[2] = np.array(
        [
            low_speed + 0.05 * feat_std[0],
            accel_mean,
            low_jitter + 0.05 * feat_std[2],
        ]
    )

    # Shared-ish full covariances initialized from global covariance, regularized
    cov = np.cov(X.T) + 1e-2 * np.eye(3)
    covars = np.stack([cov.copy() for _ in range(cfg.N_STATES)], axis=0)
    return means, covars


def _renormalize_rows(A: np.ndarray) -> np.ndarray:
    """Row-normalize a transition matrix; repair all-zero rows with self-loops."""
    A = np.asarray(A, dtype=np.float64).copy()
    row_sums = A.sum(axis=1)
    for i, s in enumerate(row_sums):
        if s <= 0:
            A[i, :] = 0.0
            A[i, i] = 1.0
    row_sums = A.sum(axis=1, keepdims=True)
    return A / row_sums


def _enforce_stop_topology(model: GaussianHMM, remap: Dict[int, int]) -> None:
    """Zero direct transitions between the two semantic stop states."""
    inv = {v: k for k, v in remap.items()}
    s_inv, s_vol = inv[1], inv[2]
    A = model.transmat_.copy()
    A[s_inv, s_vol] = 0.0
    A[s_vol, s_inv] = 0.0
    model.transmat_ = _renormalize_rows(A)


def train_gaussian_hmm(
    train_observations: Sequence[np.ndarray],
    random_state: int = cfg.HMM_RANDOM_STATE,
) -> Tuple[GaussianHMM, Dict[int, int]]:
    """
    Fit a constrained 3-state GaussianHMM on training sequences.

    Returns ``(model, remap)`` where ``remap`` maps model state index → semantic
    label index {0: Driving, 1: Involuntary, 2: Voluntary}.
    """
    if not train_observations:
        raise ValueError("No training observations for HMM.")

    lengths = [len(o) for o in train_observations]
    X = np.vstack(train_observations)

    model = GaussianHMM(
        n_components=cfg.N_STATES,
        covariance_type=cfg.HMM_COVARIANCE_TYPE,
        n_iter=cfg.HMM_N_ITER,
        tol=cfg.HMM_TOL,
        random_state=random_state,
        init_params="",  # we set startprob / transmat / means / covars ourselves
        params="stmc",  # zero-initialized A_12/A_21 stay zero under Baum–Welch
        min_covar=1e-3,
        verbose=False,
    )

    model.startprob_ = np.array([0.70, 0.15, 0.15], dtype=np.float64)
    model.transmat_ = _init_transition_matrix()
    means, covars = _init_means_covars(X, lengths)
    model.means_ = means
    model.covars_ = covars

    model.fit(X, lengths)

    # Numerical hygiene: re-assert exact zeros on the initialized stop pair.
    model.transmat_[1, 2] = 0.0
    model.transmat_[2, 1] = 0.0
    model.transmat_ = _renormalize_rows(model.transmat_)

    remap = resolve_state_mapping(model)
    _enforce_stop_topology(model, remap)
    return model, remap


def resolve_state_mapping(model: GaussianHMM) -> Dict[int, int]:
    """
    Map learned indices → semantic labels:
      - highest mean speed (feature 0) → Driving (0)
      - among remaining, lower self-transition → Involuntary (1)
      - higher self-transition → Voluntary (2)
    """
    speed_means = model.means_[:, 0]
    driving = int(np.argmax(speed_means))
    remaining = [i for i in range(cfg.N_STATES) if i != driving]
    self_p = {i: float(model.transmat_[i, i]) for i in remaining}
    # shorter expected dwell (lower A_ii) → involuntary
    involuntary = min(remaining, key=lambda i: self_p[i])
    voluntary = max(remaining, key=lambda i: self_p[i])
    # model_state -> semantic
    return {driving: 0, involuntary: 1, voluntary: 2}


def apply_state_remap(states: np.ndarray, remap: Dict[int, int]) -> np.ndarray:
    out = np.empty_like(states)
    for src, dst in remap.items():
        out[states == src] = dst
    return out


def decode_segment(
    O_scaled: np.ndarray,
    model: GaussianHMM,
    remap: Dict[int, int],
) -> np.ndarray:
    """Viterbi decode one segment; returns semantic state indices."""
    # hmmlearn.decode returns (logprob, state_sequence)
    _logprob, raw = model.decode(O_scaled, algorithm="viterbi")
    return apply_state_remap(np.asarray(raw, dtype=np.int64), remap)


def backtransform_emission_stats(
    model: GaussianHMM,
    scaler: StandardScaler,
    remap: Dict[int, int],
) -> Dict[str, dict]:
    """
    Convert learned emission means/variances from standardized space back to
    raw feature space (log-speed, accel, log-jitter).
    """
    inv_remap = {v: k for k, v in remap.items()}  # semantic -> model
    means_std = scaler.mean_
    scales = scaler.scale_

    result: Dict[str, dict] = {}
    for sem in range(cfg.N_STATES):
        m_idx = inv_remap[sem]
        mu_std = model.means_[m_idx]
        cov_std = model.covars_[m_idx]
        mu_raw = mu_std * scales + means_std
        # Var(raw_i) = scale_i^2 * Var(std_i)
        var_raw = np.diag(cov_std) * (scales**2)
        label = cfg.STATE_LABELS[sem]
        result[label] = {
            "mean_log_speed": float(mu_raw[0]),
            "mean_accel": float(mu_raw[1]),
            "mean_log_jitter": float(mu_raw[2]),
            "var_log_speed": float(var_raw[0]),
            "var_accel": float(var_raw[1]),
            "var_log_jitter": float(var_raw[2]),
            "mean_speed_approx_m_s": float(max(0.0, np.exp(mu_raw[0]) - cfg.SPEED_LOG_EPS)),
            "mean_jitter_approx_m": float(max(0.0, np.exp(mu_raw[2]) - cfg.JITTER_LOG_EPS)),
            "model_state_index": int(m_idx),
        }
    return result


def semantic_transition_matrix(
    model: GaussianHMM,
    remap: Dict[int, int],
) -> np.ndarray:
    """Reorder ``transmat_`` into semantic state order [Driving, Invol, Vol]."""
    inv = {v: k for k, v in remap.items()}
    A = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=np.float64)
    for i in range(cfg.N_STATES):
        for j in range(cfg.N_STATES):
            A[i, j] = model.transmat_[inv[i], inv[j]]
    return A
