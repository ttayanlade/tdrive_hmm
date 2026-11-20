"""Run merging, quality flags, parquet exports, and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import config as cfg

PathLike = Union[str, Path]


def _state_label(state: int) -> str:
    return cfg.STATE_LABELS[int(state)]


def merge_state_runs(seg: pd.DataFrame) -> List[dict]:
    """
    Merge contiguous identical decoded states into runs.

    Expects columns: taxi_id, segment_id, timestamp, longitude, latitude,
    decoded_state, bridged_after_s, and optionally x_hat/y_hat for centroids.
    """
    if seg.empty or "decoded_state" not in seg.columns:
        return []

    states = seg["decoded_state"].to_numpy()
    # run-length encode
    change = np.flatnonzero(np.diff(states)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(states)]  # exclusive

    lon_col = "longitude"
    lat_col = "latitude"
    runs: List[dict] = []
    for s, e in zip(starts, ends):
        block = seg.iloc[s:e]
        state = int(states[s])
        label = _state_label(state)
        t0 = block["timestamp"].iloc[0]
        t1 = block["timestamp"].iloc[-1]
        observed = float((t1 - t0).total_seconds())

        bridged_vals = block["bridged_after_s"].to_numpy(dtype=np.float64)
        # Include bridged dwell on all points in the run (incl. final stop point)
        bridged = float(np.nansum(bridged_vals))
        total = observed + bridged

        runs.append(
            {
                "taxi_id": int(block["taxi_id"].iloc[0]),
                "segment_id": int(block["segment_id"].iloc[0]),
                "run_start_idx": int(s),
                "run_end_idx": int(e - 1),
                "centroid_lon": float(block[lon_col].mean()),
                "centroid_lat": float(block[lat_col].mean()),
                "start_time": t0,
                "end_time": t1,
                "observed_duration_s": observed,
                "bridged_duration_s": bridged,
                "total_duration_s": total,
                "state": state,
                "state_label": label,
                "n_points": int(e - s),
            }
        )
    return runs


def assign_stop_flag(state_label: str, total_duration_s: float) -> Optional[str]:
    """
    Non-overlapping quality/ambiguity flags for stop runs.

    Driving runs receive ``None``.
    """
    if state_label == cfg.STATE_DRIVING:
        return None
    if state_label == cfg.STATE_INVOLUNTARY:
        if total_duration_s < 10.0:
            return "too_short_lt_10s"
        if total_duration_s > 120.0:
            return "exceeds_120s_check_voluntary"
        return "valid_traffic_stop_window"
    if state_label == cfg.STATE_VOLUNTARY:
        if total_duration_s <= 120.0:
            return "short_ambiguous_le_120s"
        return "valid_voluntary_stop_window"
    return None


def build_stop_events(runs: Sequence[dict]) -> pd.DataFrame:
    """Filter to stop-state runs and attach quality flags."""
    rows = []
    for r in runs:
        if r["state_label"] == cfg.STATE_DRIVING:
            continue
        flag = assign_stop_flag(r["state_label"], r["total_duration_s"])
        rows.append(
            {
                "taxi_id": r["taxi_id"],
                "segment_id": r["segment_id"],
                "centroid_lon": r["centroid_lon"],
                "centroid_lat": r["centroid_lat"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "observed_duration_s": r["observed_duration_s"],
                "bridged_duration_s": r["bridged_duration_s"],
                "total_duration_s": r["total_duration_s"],
                "state_label": r["state_label"],
                "flag": flag,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "taxi_id",
                "segment_id",
                "centroid_lon",
                "centroid_lat",
                "start_time",
                "end_time",
                "observed_duration_s",
                "bridged_duration_s",
                "total_duration_s",
                "state_label",
                "flag",
            ]
        )
    return pd.DataFrame(rows)


def points_table_from_segments(segments: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Stack decoded segments into the point-level export schema."""
    cols = [
        "taxi_id",
        "segment_id",
        "timestamp",
        "longitude",
        "latitude",
        "smoothed_speed",
        "smoothed_accel",
        "jitter_radius",
        "decoded_state",
    ]
    frames = []
    for seg in segments:
        part = seg[cols].copy()
        part["decoded_state_label"] = part["decoded_state"].map(cfg.STATE_LABELS)
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=cols + ["decoded_state_label"])
    return pd.concat(frames, ignore_index=True)


def summarize_test_diagnostics(
    points: pd.DataFrame,
    stops: pd.DataFrame,
    test_taxi_ids: Iterable[int],
) -> Dict[str, Any]:
    """Summary counts and duration percentiles on the held-out taxi set."""
    test_ids = set(int(t) for t in test_taxi_ids)
    pts = points[points["taxi_id"].isin(test_ids)] if not points.empty else points
    st = stops[stops["taxi_id"].isin(test_ids)] if not stops.empty else stops

    by_state: Dict[str, Any] = {}
    if not pts.empty:
        for state, label in cfg.STATE_LABELS.items():
            mask = pts["decoded_state"] == state
            by_state[label] = {
                "n_points": int(mask.sum()),
                "n_taxis": int(pts.loc[mask, "taxi_id"].nunique()) if mask.any() else 0,
            }

    by_flag: Dict[str, Any] = {}
    duration_pct: Dict[str, Any] = {}
    if not st.empty:
        for (label, flag), grp in st.groupby(["state_label", "flag"], dropna=False):
            key = f"{label}::{flag}"
            durs = grp["total_duration_s"].to_numpy(dtype=np.float64)
            by_flag[key] = {
                "n_events": int(len(grp)),
                "total_duration_s": float(np.nansum(durs)),
            }
            duration_pct[key] = {
                "p50": float(np.nanpercentile(durs, 50)),
                "p90": float(np.nanpercentile(durs, 90)),
                "p99": float(np.nanpercentile(durs, 99)),
                "mean": float(np.nanmean(durs)),
            }

    return {
        "n_test_taxis": len(test_ids),
        "n_test_points": int(len(pts)),
        "n_test_stop_events": int(len(st)),
        "points_by_state": by_state,
        "stop_events_by_flag": by_flag,
        "duration_percentiles_by_flag": duration_pct,
    }


def write_outputs(
    output_dir: PathLike,
    points: pd.DataFrame,
    stops: pd.DataFrame,
    diagnostics: Dict[str, Any],
) -> Dict[str, Path]:
    """Write parquet + JSON deliverables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = output_dir / "decoded_points.parquet"
    stops_path = output_dir / "stop_events.parquet"
    diag_path = output_dir / "model_diagnostics.json"

    points.to_parquet(points_path, index=False)
    stops.to_parquet(stops_path, index=False)
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    return {
        "decoded_points": points_path,
        "stop_events": stops_path,
        "model_diagnostics": diag_path,
    }
