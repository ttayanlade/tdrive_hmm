#!/usr/bin/env python3
"""End-to-end T-Drive 3-state Gaussian HMM pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from . import config as cfg
from .model import (
    backtransform_emission_stats,
    decode_segment,
    enrich_segment_features,
    fit_scaler,
    scale_segment,
    semantic_transition_matrix,
    train_gaussian_hmm,
)
from .postprocess import (
    build_stop_events,
    merge_state_runs,
    points_table_from_segments,
    summarize_test_diagnostics,
    write_outputs,
)
from .preprocessing import process_all_taxis, split_taxi_ids
from .smoothing import smooth_segment_record

logger = logging.getLogger("tdrive_hmm")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _smooth_one(rec: dict) -> dict:
    return smooth_segment_record(rec)


def _feature_one(rec: dict) -> pd.DataFrame:
    from .preprocessing import record_to_dataframe
    from .smoothing import smooth_segment

    # Prefer already-smoothed fields when present
    if "smoothed_speed" in rec:
        seg = record_to_dataframe(rec)
        seg["x_hat"] = rec["x_hat"]
        seg["y_hat"] = rec["y_hat"]
        seg["smoothed_speed"] = rec["smoothed_speed"]
        seg["smoothed_accel"] = rec["smoothed_accel"]
    else:
        seg = smooth_segment(record_to_dataframe(rec))
    return enrich_segment_features(seg)


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    n_jobs: int = -1,
    max_taxis: Optional[int] = None,
    seed: int = cfg.RANDOM_SEED,
) -> Dict[str, Path]:
    """
    Full pipeline: preprocess → smooth → features → HMM → decode → export.
    """
    logger.info("Step 1/5 — preprocessing taxi files from %s", data_dir)
    segments = process_all_taxis(
        data_dir=data_dir,
        n_jobs=n_jobs,
        max_taxis=max_taxis,
        show_progress=True,
        seed=seed,
    )
    if not segments:
        raise RuntimeError(
            "No high-frequency segments survived cleaning. "
            "Check data path / bbox / sampling filters."
        )
    logger.info("Kept %d high-frequency segments", len(segments))

    taxi_ids = sorted({int(s["taxi_id"]) for s in segments})
    train_ids, test_ids = split_taxi_ids(taxi_ids, seed=seed)
    train_set = set(int(t) for t in train_ids)
    test_set = set(int(t) for t in test_ids)
    if len(taxi_ids) < 5:
        logger.warning(
            "Only %d taxi(s) yielded HF segments; train/test diagnostics will be thin. "
            "Increase --max-taxis or use the full corpus.",
            len(taxi_ids),
        )
    logger.info(
        "Taxi split: %d train / %d test (seed=%d)",
        len(train_set),
        len(test_set),
        seed,
    )

    logger.info("Step 2/5 — Kalman + RTS smoothing")
    smoothed: List[dict] = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_smooth_one)(rec)
        for rec in tqdm(segments, desc="Smooth segments")
    )

    logger.info("Step 3/5 — feature engineering + StandardScaler (train only)")
    featured: List[pd.DataFrame] = []
    for rec in tqdm(smoothed, desc="Features"):
        featured.append(_feature_one(rec))

    train_segs = [s for s in featured if int(s["taxi_id"].iloc[0]) in train_set]
    if not train_segs:
        raise RuntimeError("No training segments after taxi-level split.")

    scaler = fit_scaler(train_segs)

    scaled_segs: List[pd.DataFrame] = []
    train_obs: List[np.ndarray] = []
    for seg in tqdm(featured, desc="Scale features"):
        scaled, O = scale_segment(seg, scaler)
        scaled_segs.append(scaled)
        if int(scaled["taxi_id"].iloc[0]) in train_set:
            train_obs.append(O)

    logger.info("Step 4/5 — train constrained 3-state GaussianHMM")
    model, remap = train_gaussian_hmm(train_obs, random_state=seed)
    logger.info("State remap (model→semantic): %s", remap)
    logger.info("Transition matrix (semantic order):\n%s", semantic_transition_matrix(model, remap))

    logger.info("Step 5/5 — Viterbi decode, merge runs, export")
    decoded: List[pd.DataFrame] = []
    all_runs: List[dict] = []
    for seg in tqdm(scaled_segs, desc="Decode"):
        O = np.column_stack([seg["o0"], seg["o1"], seg["o2"]])
        states = decode_segment(O, model, remap)
        out = seg.copy()
        out["decoded_state"] = states
        decoded.append(out)
        all_runs.extend(merge_state_runs(out))

    points_int = points_table_from_segments(decoded)
    points = points_int.rename(columns={"decoded_state": "decoded_state_id"})
    points["decoded_state"] = points["decoded_state_label"]
    points = points[
        [
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
    ]

    stops = build_stop_events(all_runs)
    test_summary = summarize_test_diagnostics(points_int, stops, test_ids)

    A_sem = semantic_transition_matrix(model, remap)
    emissions = backtransform_emission_stats(model, scaler, remap)

    diagnostics = {
        "n_taxis_total": len(taxi_ids),
        "n_taxis_train": len(train_set),
        "n_taxis_test": len(test_set),
        "n_segments": len(decoded),
        "train_taxi_ids_sample": sorted(list(train_set))[:20],
        "test_taxi_ids_sample": sorted(list(test_set))[:20],
        "random_seed": seed,
        "hmm": {
            "n_states": cfg.N_STATES,
            "covariance_type": cfg.HMM_COVARIANCE_TYPE,
            "n_iter": int(getattr(model, "n_iter", cfg.HMM_N_ITER)),
            "converged": bool(getattr(model.monitor_, "converged", False)),
            "state_remap_model_to_semantic": {str(k): int(v) for k, v in remap.items()},
            "transition_matrix_semantic_order": {
                "order": [cfg.STATE_LABELS[i] for i in range(cfg.N_STATES)],
                "matrix": A_sem.tolist(),
            },
            "startprob_semantic": [
                float(model.startprob_[{v: k for k, v in remap.items()}[i]])
                for i in range(cfg.N_STATES)
            ],
            "emissions_backtransformed": emissions,
        },
        "test_set_summary": test_summary,
        "config": {
            "gap_threshold_s": cfg.GAP_THRESHOLD_S,
            "bridge_max_displacement_m": cfg.BRIDGE_MAX_DISPLACEMENT_M,
            "high_freq_median_dt_s": cfg.HIGH_FREQ_MEDIAN_DT_S,
            "min_segment_points": cfg.MIN_SEGMENT_POINTS,
            "gps_sigma_m": cfg.GPS_SIGMA_M,
            "process_accel_sigma": cfg.PROCESS_ACCEL_SIGMA,
            "jitter_window_s": cfg.JITTER_WINDOW_S,
        },
    }

    paths = write_outputs(output_dir, points, stops, diagnostics)
    logger.info("Wrote %s", paths["decoded_points"])
    logger.info("Wrote %s", paths["stop_events"])
    logger.info("Wrote %s", paths["model_diagnostics"])
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-State Gaussian HMM for T-Drive trajectory stop classification",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory of per-taxi .txt files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for parquet/JSON deliverables",
    )
    p.add_argument("--n-jobs", type=int, default=-1, help="joblib parallelism")
    p.add_argument(
        "--max-taxis",
        type=int,
        default=None,
        help="Optional cap on number of taxi files (smoke / debug)",
    )
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        return 1
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        max_taxis=args.max_taxis,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
