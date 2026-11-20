"""Data cleaning, metric projection, cross-gap bridging, and HF segment extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyproj import Transformer
from tqdm import tqdm

from . import config as cfg

PathLike = Union[str, Path]


def _transformer() -> Transformer:
    return Transformer.from_crs("EPSG:4326", cfg.UTM_EPSG, always_xy=True)


def list_taxi_files(data_dir: PathLike) -> List[Path]:
    """Return sorted list of per-taxi ``.txt`` files."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")
    return files


def load_taxi_file(path: PathLike) -> pd.DataFrame:
    """Load a single T-Drive taxi file into a DataFrame."""
    path = Path(path)
    df = pd.read_csv(
        path,
        header=None,
        names=cfg.RAW_COLUMNS,
        dtype={"taxi_id": "int64", "longitude": "float64", "latitude": "float64"},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp", "longitude", "latitude"])
    return df


def _drop_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp", kind="mergesort")
    return df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)


def _filter_beijing_bbox(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["latitude"] >= cfg.BEIJING_LAT_MIN)
        & (df["latitude"] <= cfg.BEIJING_LAT_MAX)
        & (df["longitude"] >= cfg.BEIJING_LON_MIN)
        & (df["longitude"] <= cfg.BEIJING_LON_MAX)
    )
    return df.loc[mask].reset_index(drop=True)


def _project_to_utm(df: pd.DataFrame, transformer: Optional[Transformer] = None) -> pd.DataFrame:
    transformer = transformer or _transformer()
    x, y = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    out = df.copy()
    out["x"] = np.asarray(x, dtype=np.float64)
    out["y"] = np.asarray(y, dtype=np.float64)
    return out


def _timestamps_to_seconds(ts: pd.Series) -> np.ndarray:
    """Convert pandas timestamps to Unix seconds (unit-agnostic)."""
    # Normalize to ns to avoid datetime64[us] vs [ns] surprises across pandas builds.
    ns = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return ns.astype(np.float64) / 1e9


def _compute_steps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t_s"] = _timestamps_to_seconds(out["timestamp"])
    dx = out["x"].diff()
    dy = out["y"].diff()
    out["delta_t"] = out["t_s"].diff()
    out["step_dist"] = np.sqrt(dx.to_numpy(dtype=np.float64) ** 2 + dy.to_numpy(dtype=np.float64) ** 2)
    return out


def _filter_teleports(df: pd.DataFrame, max_speed: float = cfg.MAX_RAW_SPEED_M_S) -> pd.DataFrame:
    """Drop destination points of jumps whose implied speed exceeds ``max_speed``."""
    if len(df) < 2:
        return df.reset_index(drop=True)

    keep = np.ones(len(df), dtype=bool)
    # Iteratively remove teleport destinations; rare but handles chains of bad points.
    changed = True
    work = df
    while changed and len(work) >= 2:
        work = _compute_steps(work)
        dt = work["delta_t"].to_numpy()
        dist = work["step_dist"].to_numpy()
        speed = np.full(len(work), np.nan)
        valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dist)
        speed[valid] = dist[valid] / dt[valid]
        drop = np.zeros(len(work), dtype=bool)
        drop[1:] = np.isfinite(speed[1:]) & (speed[1:] > max_speed)
        if not drop.any():
            changed = False
            break
        work = work.loc[~drop].drop(columns=["t_s", "delta_t", "step_dist"], errors="ignore")
        work = work.reset_index(drop=True)
        changed = True

    return work.reset_index(drop=True)


def _apply_cross_gap_bridging(df: pd.DataFrame) -> pd.DataFrame:
    """
    Before gap-splitting, attach standby dwell to the pre-gap point when
    ``delta_t > G`` and displacement ``<= BRIDGE_MAX_DISPLACEMENT_M``.
    """
    out = _compute_steps(df)
    bridged = np.zeros(len(out), dtype=np.float64)
    dt = out["delta_t"].to_numpy()
    dist = out["step_dist"].to_numpy()
    for i in range(1, len(out)):
        if (
            np.isfinite(dt[i])
            and dt[i] > cfg.GAP_THRESHOLD_S
            and np.isfinite(dist[i])
            and dist[i] <= cfg.BRIDGE_MAX_DISPLACEMENT_M
        ):
            bridged[i - 1] = float(dt[i])
    out["bridged_after_s"] = bridged
    return out


def _split_on_gaps(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split wherever ``delta_t > GAP_THRESHOLD_S`` (bridged info already recorded)."""
    if len(df) == 0:
        return []
    dt = df["delta_t"].to_numpy()
    break_idx = np.where(np.isfinite(dt) & (dt > cfg.GAP_THRESHOLD_S))[0]
    # break at index i means start a new segment at i
    starts = [0] + break_idx.tolist()
    ends = break_idx.tolist() + [len(df)]
    segments: List[pd.DataFrame] = []
    for s, e in zip(starts, ends):
        seg = df.iloc[s:e].copy()
        if len(seg) == 0:
            continue
        bridged = (
            seg["bridged_after_s"].to_numpy(dtype=np.float64)
            if "bridged_after_s" in seg.columns
            else np.zeros(len(seg), dtype=np.float64)
        )
        # Recompute intra-segment deltas for HF filters (first row NaN)
        seg = _compute_steps(seg.drop(columns=["t_s", "delta_t", "step_dist", "bridged_after_s"], errors="ignore"))
        seg["bridged_after_s"] = bridged
        segments.append(seg.reset_index(drop=True))
    return segments


def _is_high_frequency_segment(seg: pd.DataFrame) -> bool:
    if len(seg) < cfg.MIN_SEGMENT_POINTS:
        return False
    dt = seg["delta_t"].to_numpy()[1:]  # skip first NaN
    if len(dt) == 0 or not np.isfinite(dt).any():
        return False
    median_dt = float(np.nanmedian(dt))
    if median_dt > cfg.HIGH_FREQ_MEDIAN_DT_S:
        return False
    span = float(seg["t_s"].iloc[-1] - seg["t_s"].iloc[0])
    if span < cfg.MIN_SEGMENT_SPAN_S:
        return False
    return True


def process_taxi_dataframe(
    df: pd.DataFrame,
    transformer: Optional[Transformer] = None,
) -> List[pd.DataFrame]:
    """
    Clean one taxi trajectory and return high-frequency segments.

    Each returned segment has columns:
    ``taxi_id, timestamp, longitude, latitude, x, y, t_s, delta_t, step_dist, bridged_after_s``.
    """
    if df.empty:
        return []

    df = _drop_duplicate_timestamps(df)
    df = _filter_beijing_bbox(df)
    if len(df) < cfg.MIN_SEGMENT_POINTS:
        return []

    df = _project_to_utm(df, transformer=transformer)
    df = _filter_teleports(df)
    if len(df) < cfg.MIN_SEGMENT_POINTS:
        return []

    df = _apply_cross_gap_bridging(df)
    raw_segments = _split_on_gaps(df)

    # Re-attach bridged_after_s after recompute in _split_on_gaps:
    # _split_on_gaps recomputes steps but should keep bridged_after_s.
    # Ensure first-point bridged values that belonged to gap starts are preserved.
    hf_segments: List[pd.DataFrame] = []
    for seg in raw_segments:
        if "bridged_after_s" not in seg.columns:
            seg["bridged_after_s"] = 0.0
        else:
            seg["bridged_after_s"] = seg["bridged_after_s"].fillna(0.0).astype(np.float64)
        if _is_high_frequency_segment(seg):
            # Clear bridged on last point of segment if it was for a gap that caused the split
            # (that bridge was attached to pre-gap point which is the last point of this segment —
            # keep it: voluntary stop preservation requires it on the stop run's final point).
            hf_segments.append(seg.reset_index(drop=True))
    return hf_segments


def process_taxi_file(
    path: PathLike,
    transformer: Optional[Transformer] = None,
) -> List[pd.DataFrame]:
    """Load and process a single taxi file into HF segments."""
    df = load_taxi_file(path)
    return process_taxi_dataframe(df, transformer=transformer)


def _process_one_file(path: str) -> List[dict]:
    """Worker: return serializable segment records (dicts of arrays / lists)."""
    transformer = _transformer()
    segments = process_taxi_file(path, transformer=transformer)
    records: List[dict] = []
    taxi_id = Path(path).stem
    for i, seg in enumerate(segments):
        records.append(
            {
                "taxi_id": int(seg["taxi_id"].iloc[0]) if "taxi_id" in seg.columns else int(taxi_id),
                "segment_id": i,
                "timestamp": seg["timestamp"].to_numpy(),
                "longitude": seg["longitude"].to_numpy(dtype=np.float64),
                "latitude": seg["latitude"].to_numpy(dtype=np.float64),
                "x": seg["x"].to_numpy(dtype=np.float64),
                "y": seg["y"].to_numpy(dtype=np.float64),
                "t_s": seg["t_s"].to_numpy(dtype=np.float64),
                "delta_t": seg["delta_t"].to_numpy(dtype=np.float64),
                "bridged_after_s": seg["bridged_after_s"].to_numpy(dtype=np.float64),
            }
        )
    return records


def record_to_dataframe(rec: dict) -> pd.DataFrame:
    """Convert a segment record dict back to a DataFrame."""
    n = len(rec["timestamp"])
    return pd.DataFrame(
        {
            "taxi_id": np.full(n, rec["taxi_id"], dtype=np.int64),
            "segment_id": np.full(n, rec["segment_id"], dtype=np.int64),
            "timestamp": pd.to_datetime(rec["timestamp"]),
            "longitude": rec["longitude"],
            "latitude": rec["latitude"],
            "x": rec["x"],
            "y": rec["y"],
            "t_s": rec["t_s"],
            "delta_t": rec["delta_t"],
            "bridged_after_s": rec["bridged_after_s"],
        }
    )


def process_all_taxis(
    data_dir: PathLike,
    n_jobs: int = -1,
    max_taxis: Optional[int] = None,
    taxi_ids: Optional[Sequence[int]] = None,
    show_progress: bool = True,
    seed: int = cfg.RANDOM_SEED,
) -> List[dict]:
    """
    Process all (or a subset of) taxi files in parallel.

    When ``max_taxis`` is set (and ``taxi_ids`` is not), files are shuffled with
    ``seed`` before truncation so smoke runs are not biased to lexicographic IDs.

    Returns a list of segment record dicts suitable for downstream smoothing.
    """
    files = list_taxi_files(data_dir)
    if taxi_ids is not None:
        wanted = {str(t) for t in taxi_ids}
        files = [f for f in files if f.stem in wanted]
    elif max_taxis is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        files = [files[i] for i in order[: int(max_taxis)]]

    paths = [str(f) for f in files]
    batches = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_one_file)(p)
        for p in tqdm(paths, desc="Preprocess taxis", disable=not show_progress)
    )
    all_segments: List[dict] = []
    for batch in batches:
        all_segments.extend(batch)
    return all_segments


def split_taxi_ids(
    taxi_ids: Iterable[int],
    train_fraction: float = cfg.TRAIN_FRACTION,
    seed: int = cfg.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split unique taxi IDs into train / test sets (no trajectory leakage)."""
    ids = np.array(sorted(set(int(t) for t in taxi_ids)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(round(len(ids) * train_fraction))
    n_train = min(max(n_train, 1 if len(ids) > 1 else len(ids)), len(ids))
    if len(ids) > 1 and n_train == len(ids):
        n_train = len(ids) - 1
    train_ids = ids[:n_train]
    test_ids = ids[n_train:]
    return train_ids, test_ids
