"""Unit tests with synthetic high-frequency trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdrive_hmm import config as cfg
from tdrive_hmm.model import (
    build_observation_matrix,
    enrich_segment_features,
    fit_scaler,
    resolve_state_mapping,
    scale_segment,
    train_gaussian_hmm,
)
from tdrive_hmm.postprocess import assign_stop_flag, build_stop_events, merge_state_runs
from tdrive_hmm.preprocessing import process_taxi_dataframe, split_taxi_ids
from tdrive_hmm.smoothing import kalman_rts_smooth_2d, smooth_segment


def _synthetic_track(
    taxi_id: int = 1,
    n: int = 80,
    dt: float = 5.0,
    mode: str = "drive",
    start_lon: float = 116.4,
    start_lat: float = 39.9,
) -> pd.DataFrame:
    """Build a Beijing-local track with ~dt second sampling."""
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2008-02-02 12:00:00")
    timestamps = [t0 + pd.Timedelta(seconds=i * dt) for i in range(n)]
    # ~ meters to degrees (approx at Beijing)
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * np.cos(np.deg2rad(start_lat))

    lons = np.zeros(n)
    lats = np.zeros(n)
    lons[0], lats[0] = start_lon, start_lat
    for i in range(1, n):
        if mode == "drive":
            speed = 12.0 + rng.normal(0, 0.5)  # m/s
            heading = 0.3
            dx = speed * dt * np.cos(heading)
            dy = speed * dt * np.sin(heading)
        elif mode == "stop":
            dx = rng.normal(0, 1.5)
            dy = rng.normal(0, 1.5)
        else:
            raise ValueError(mode)
        lons[i] = lons[i - 1] + dx / m_per_deg_lon
        lats[i] = lats[i - 1] + dy / m_per_deg_lat

    return pd.DataFrame(
        {
            "taxi_id": taxi_id,
            "timestamp": timestamps,
            "longitude": lons,
            "latitude": lats,
        }
    )


def test_cross_gap_bridging_preserves_stationary_standby():
    # HF drive, then park (tiny displacement) with 300s standby gap, then resume
    a = _synthetic_track(n=30, dt=5.0, mode="drive")
    parked = a.iloc[[-1]].copy()
    parked["timestamp"] = a["timestamp"].iloc[-1] + pd.Timedelta(seconds=300)
    # ~10 m nudge in lon
    parked["longitude"] = a["longitude"].iloc[-1] + 10.0 / (111_000.0 * np.cos(np.deg2rad(39.9)))
    b = _synthetic_track(n=30, dt=5.0, mode="drive", start_lon=float(parked["longitude"].iloc[0]), start_lat=float(parked["latitude"].iloc[0]))
    b["timestamp"] = [
        parked["timestamp"].iloc[0] + pd.Timedelta(seconds=5 * (i + 1)) for i in range(len(b))
    ]
    df = pd.concat([a, parked, b], ignore_index=True)
    segs = process_taxi_dataframe(df)
    assert len(segs) >= 2
    # Pre-gap point (end of first HF segment) should carry bridged_after_s ≈ 300
    first = segs[0]
    assert first["bridged_after_s"].iloc[-1] >= 250.0


def test_hf_segment_filter_and_smoothing():
    df = _synthetic_track(n=40, dt=5.0, mode="drive")
    segs = process_taxi_dataframe(df)
    assert len(segs) == 1
    sm = smooth_segment(segs[0])
    assert "smoothed_speed" in sm.columns
    assert np.all(np.isfinite(sm["smoothed_speed"]))
    assert sm["smoothed_accel"].between(cfg.ACCEL_CLIP_MIN, cfg.ACCEL_CLIP_MAX).all()


def test_kalman_reduces_jitter_on_static():
    df = _synthetic_track(n=40, dt=5.0, mode="stop")
    segs = process_taxi_dataframe(df)
    assert segs
    seg = segs[0]
    x_hat, y_hat, *_ = kalman_rts_smooth_2d(
        seg["x"].to_numpy(), seg["y"].to_numpy(), seg["t_s"].to_numpy()
    )
    raw_r = np.sqrt((seg["x"] - seg["x"].mean()) ** 2 + (seg["y"] - seg["y"].mean()) ** 2).mean()
    sm_r = np.sqrt((x_hat - x_hat.mean()) ** 2 + (y_hat - y_hat.mean()) ** 2).mean()
    assert sm_r <= raw_r + 1e-6


def test_hmm_train_decode_and_flags():
    # Multiple taxis with clearly separable drive + short stop + long stop
    records = []
    for tid in range(1, 9):
        drive = _synthetic_track(taxi_id=tid, n=60, dt=5.0, mode="drive")
        short_stop = _synthetic_track(
            taxi_id=tid,
            n=8,
            dt=5.0,
            mode="stop",
            start_lon=float(drive["longitude"].iloc[-1]),
            start_lat=float(drive["latitude"].iloc[-1]),
        )
        short_stop["timestamp"] = [
            drive["timestamp"].iloc[-1] + pd.Timedelta(seconds=5 * (i + 1)) for i in range(len(short_stop))
        ]
        drive2 = _synthetic_track(
            taxi_id=tid,
            n=40,
            dt=5.0,
            mode="drive",
            start_lon=float(short_stop["longitude"].iloc[-1]) + 0.001,
            start_lat=float(short_stop["latitude"].iloc[-1]),
        )
        drive2["timestamp"] = [
            short_stop["timestamp"].iloc[-1] + pd.Timedelta(seconds=5 * (i + 1)) for i in range(len(drive2))
        ]
        long_stop = _synthetic_track(
            taxi_id=tid,
            n=50,
            dt=5.0,
            mode="stop",
            start_lon=float(drive2["longitude"].iloc[-1]),
            start_lat=float(drive2["latitude"].iloc[-1]),
        )
        long_stop["timestamp"] = [
            drive2["timestamp"].iloc[-1] + pd.Timedelta(seconds=5 * (i + 1)) for i in range(len(long_stop))
        ]
        df = pd.concat([drive, short_stop, drive2, long_stop], ignore_index=True)
        segs = process_taxi_dataframe(df)
        for s in segs:
            s["segment_id"] = 0
            records.append(smooth_segment(s))

    assert records, "expected synthetic HF segments"
    featured = [enrich_segment_features(s) for s in records]
    train_ids, test_ids = split_taxi_ids([s["taxi_id"].iloc[0] for s in featured], seed=42)
    train_set = set(map(int, train_ids))
    train_segs = [s for s in featured if int(s["taxi_id"].iloc[0]) in train_set]
    scaler = fit_scaler(train_segs)
    train_obs = []
    scaled = []
    for s in featured:
        sc, O = scale_segment(s, scaler)
        scaled.append(sc)
        if int(sc["taxi_id"].iloc[0]) in train_set:
            train_obs.append(O)

    model, remap = train_gaussian_hmm(train_obs, random_state=42)
    assert set(remap.values()) == {0, 1, 2}
    # Constraints preserved between semantic stop states
    inv = {v: k for k, v in remap.items()}
    assert model.transmat_[inv[1], inv[2]] == pytest.approx(0.0, abs=1e-12)
    assert model.transmat_[inv[2], inv[1]] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(model.transmat_.sum(axis=1), 1.0)

    # Decode + flags
    all_runs = []
    for sc in scaled:
        from tdrive_hmm.model import decode_segment

        O = np.column_stack([sc["o0"], sc["o1"], sc["o2"]])
        sc = sc.copy()
        decoded = decode_segment(O, model, remap)
        assert decoded.shape == (len(sc),)
        assert set(np.unique(decoded)).issubset({0, 1, 2})
        sc["decoded_state"] = decoded
        all_runs.extend(merge_state_runs(sc))

    stops = build_stop_events(all_runs)
    assert assign_stop_flag(cfg.STATE_INVOLUNTARY, 5.0) == "too_short_lt_10s"
    assert assign_stop_flag(cfg.STATE_INVOLUNTARY, 30.0) == "valid_traffic_stop_window"
    assert assign_stop_flag(cfg.STATE_INVOLUNTARY, 200.0) == "exceeds_120s_check_voluntary"
    assert assign_stop_flag(cfg.STATE_VOLUNTARY, 60.0) == "short_ambiguous_le_120s"
    assert assign_stop_flag(cfg.STATE_VOLUNTARY, 200.0) == "valid_voluntary_stop_window"
    assert isinstance(stops, pd.DataFrame)


def test_split_taxi_ids_reproducible():
    a, b = split_taxi_ids(range(100), seed=42)
    c, d = split_taxi_ids(range(100), seed=42)
    assert np.array_equal(a, c) and np.array_equal(b, d)
    assert len(a) + len(b) == 100
    assert abs(len(a) / 100 - 0.8) < 1e-9
