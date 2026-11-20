#!/usr/bin/env python3
"""Offline stop QC plots (no map-tile network access required)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .visualize_stops import _context_track, _filter_stops


STATE_COLORS = {
    cfg.STATE_DRIVING: "#2c7bb6",
    cfg.STATE_INVOLUNTARY: "#fdae61",
    cfg.STATE_VOLUNTARY: "#d7191c",
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline random-stop QC plots")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--fig-dir", type=Path, default=Path("outputs/stop_qc_figs"))
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", choices=["voluntary", "involuntary", "all"], default="voluntary")
    p.add_argument("--flag", default="valid_voluntary_stop_window")
    p.add_argument("--pad-s", type=float, default=180.0)
    p.add_argument("--min-duration-s", type=float, default=0.0)
    p.add_argument("--open", action="store_true", help="Open the first figure after writing")
    return p.parse_args(argv)


def _plot_one(event: pd.Series, track: pd.DataFrame, out_path: Path, idx: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    ax_map, ax_ts = axes

    if track.empty:
        ax_map.text(0.5, 0.5, "No trajectory context found", ha="center", va="center")
        ax_ts.axis("off")
    else:
        # Spatial panel (local meters via equirectangular approx)
        lat0 = float(track["latitude"].mean())
        lon0 = float(track["longitude"].mean())
        m_per_deg_lat = 111_000.0
        m_per_deg_lon = 111_000.0 * np.cos(np.deg2rad(lat0))
        x = (track["longitude"].to_numpy() - lon0) * m_per_deg_lon
        y = (track["latitude"].to_numpy() - lat0) * m_per_deg_lat
        cx = (float(event["centroid_lon"]) - lon0) * m_per_deg_lon
        cy = (float(event["centroid_lat"]) - lat0) * m_per_deg_lat

        ax_map.plot(x, y, color="#888888", lw=1.5, alpha=0.8, label="track")
        if "decoded_state" in track.columns:
            for state, color in STATE_COLORS.items():
                m = track["decoded_state"].to_numpy() == state
                if m.any():
                    ax_map.scatter(x[m], y[m], s=14, c=color, label=state, zorder=3)
        ax_map.scatter([cx], [cy], s=120, c="red", marker="x", linewidths=2, label="stop centroid", zorder=4)
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.set_xlabel("east (m)")
        ax_map.set_ylabel("north (m)")
        ax_map.grid(True, alpha=0.3)
        ax_map.legend(fontsize=8, loc="best")

        # Speed timeline
        t0 = pd.Timestamp(event["start_time"])
        t_rel = (track["timestamp"] - t0).dt.total_seconds().to_numpy()
        speed = track["smoothed_speed"].to_numpy()
        ax_ts.plot(t_rel, speed, color="#333333", lw=1.5)
        ax_ts.axvspan(0.0, float(event["observed_duration_s"]), color="#d7191c", alpha=0.2, label="observed stop")
        if float(event["bridged_duration_s"]) > 0:
            ax_ts.axvline(float(event["observed_duration_s"]), color="#d7191c", ls="--", lw=1.5)
            ax_ts.text(
                float(event["observed_duration_s"]),
                max(speed.max(), 1.0) * 0.9,
                f"bridged +{event['bridged_duration_s']:.0f}s",
                color="#d7191c",
                fontsize=8,
            )
        ax_ts.set_xlabel("seconds relative to stop start")
        ax_ts.set_ylabel("smoothed speed (m/s)")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(fontsize=8)

    title = (
        f"#{idx} taxi={event['taxi_id']} seg={event['segment_id']} | "
        f"{event['state_label']} | {event['flag']}\n"
        f"{event['start_time']} → {event['end_time']} | "
        f"total={event['total_duration_s']:.0f}s "
        f"(obs={event['observed_duration_s']:.0f}s, bridged={event['bridged_duration_s']:.0f}s)"
    )
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    stops = pd.read_parquet(args.output_dir / "stop_events.parquet")
    points = pd.read_parquet(args.output_dir / "decoded_points.parquet")
    points["timestamp"] = pd.to_datetime(points["timestamp"])

    filtered = _filter_stops(stops, args)
    n = min(int(args.n), len(filtered))
    sample = filtered.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    # clear old figures for this run prefix
    for old in args.fig_dir.glob("stop_qc_*.png"):
        old.unlink()

    paths = []
    for i, (_, event) in enumerate(sample.iterrows(), start=1):
        track = _context_track(points, event, pad_s=float(args.pad_s))
        out = args.fig_dir / f"stop_qc_{i:02d}_taxi{int(event['taxi_id'])}.png"
        _plot_one(event, track, out, idx=i)
        paths.append(out)

    index = args.fig_dir / "index.txt"
    index.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")
    print(f"Wrote {len(paths)} offline figures to {args.fig_dir.resolve()}")
    print(f"Index: {index.resolve()}")
    if args.open and paths:
        import subprocess
        import sys

        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, str(paths[0])], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
