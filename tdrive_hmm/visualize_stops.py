#!/usr/bin/env python3
"""Interactive map viewer for manual QC of inferred stop / parking events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import folium
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "folium is required for visualization. Install with:\n"
        "  pip install folium"
    ) from exc

from . import config as cfg


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Randomly sample stop events and plot trajectory context on a map",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing stop_events.parquet and decoded_points.parquet",
    )
    p.add_argument(
        "--html-out",
        type=Path,
        default=Path("outputs/stop_qc_map.html"),
        help="Path to write the interactive HTML map",
    )
    p.add_argument("--n", type=int, default=12, help="Number of random stops to show")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--label",
        choices=["voluntary", "involuntary", "all"],
        default="voluntary",
        help="Which stop class to sample",
    )
    p.add_argument(
        "--flag",
        default="valid_voluntary_stop_window",
        help=(
            "Optional flag filter. Use 'any' to disable. "
            "Defaults to valid_voluntary_stop_window for parking QC."
        ),
    )
    p.add_argument(
        "--pad-s",
        type=float,
        default=180.0,
        help="Seconds of trajectory context before/after the stop window",
    )
    p.add_argument(
        "--min-duration-s",
        type=float,
        default=0.0,
        help="Only sample stops with total_duration_s >= this",
    )
    return p.parse_args(argv)


def _filter_stops(stops: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = stops.copy()
    if args.label == "voluntary":
        out = out[out["state_label"] == cfg.STATE_VOLUNTARY]
    elif args.label == "involuntary":
        out = out[out["state_label"] == cfg.STATE_INVOLUNTARY]

    if args.flag and args.flag.lower() != "any":
        out = out[out["flag"] == args.flag]

    if args.min_duration_s > 0:
        out = out[out["total_duration_s"] >= float(args.min_duration_s)]

    if out.empty:
        raise SystemExit(
            "No stop events matched the filters. Try --flag any or --label all."
        )
    return out.reset_index(drop=True)


def _context_track(
    points: pd.DataFrame,
    event: pd.Series,
    pad_s: float,
) -> pd.DataFrame:
    """Points for the same taxi/segment, padded around the stop interval."""
    mask = (points["taxi_id"] == int(event["taxi_id"])) & (
        points["segment_id"] == int(event["segment_id"])
    )
    seg = points.loc[mask].sort_values("timestamp")
    if seg.empty:
        # Fallback: same taxi, time window only
        t0 = pd.Timestamp(event["start_time"]) - pd.Timedelta(seconds=pad_s)
        t1 = pd.Timestamp(event["end_time"]) + pd.Timedelta(seconds=pad_s)
        seg = points[
            (points["taxi_id"] == int(event["taxi_id"]))
            & (points["timestamp"] >= t0)
            & (points["timestamp"] <= t1)
        ].sort_values("timestamp")
        return seg

    t0 = pd.Timestamp(event["start_time"]) - pd.Timedelta(seconds=pad_s)
    t1 = pd.Timestamp(event["end_time"]) + pd.Timedelta(seconds=pad_s)
    return seg[(seg["timestamp"] >= t0) & (seg["timestamp"] <= t1)].copy()


STATE_COLORS = {
    cfg.STATE_DRIVING: "#2c7bb6",
    cfg.STATE_INVOLUNTARY: "#fdae61",
    cfg.STATE_VOLUNTARY: "#d7191c",
}


def _add_event_layer(
    fmap: folium.Map,
    event: pd.Series,
    track: pd.DataFrame,
    idx: int,
) -> None:
    color = STATE_COLORS.get(str(event["state_label"]), "#333333")
    popup = (
        f"<b>#{idx}</b> taxi={event['taxi_id']} seg={event['segment_id']}<br>"
        f"label={event['state_label']}<br>"
        f"flag={event['flag']}<br>"
        f"start={event['start_time']}<br>"
        f"end={event['end_time']}<br>"
        f"observed={event['observed_duration_s']:.1f}s<br>"
        f"bridged={event['bridged_duration_s']:.1f}s<br>"
        f"total={event['total_duration_s']:.1f}s"
    )

    # Trajectory polyline colored by decoded state when available
    if not track.empty:
        coords = track[["latitude", "longitude"]].to_numpy().tolist()
        folium.PolyLine(
            coords,
            color="#666666",
            weight=3,
            opacity=0.7,
            tooltip=f"taxi {event['taxi_id']} context",
        ).add_to(fmap)

        if "decoded_state" in track.columns:
            for _, row in track.iterrows():
                folium.CircleMarker(
                    location=[float(row["latitude"]), float(row["longitude"])],
                    radius=2,
                    color=STATE_COLORS.get(str(row["decoded_state"]), "#888"),
                    fill=True,
                    fill_opacity=0.8,
                    weight=0,
                ).add_to(fmap)

    folium.CircleMarker(
        location=[float(event["centroid_lat"]), float(event["centroid_lon"])],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(popup, max_width=320),
        tooltip=f"#{idx} total={event['total_duration_s']:.0f}s",
    ).add_to(fmap)


def build_qc_map(
    stops: pd.DataFrame,
    points: pd.DataFrame,
    sample: pd.DataFrame,
    pad_s: float,
) -> folium.Map:
    center_lat = float(sample["centroid_lat"].mean())
    center_lon = float(sample["centroid_lon"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # Legend
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
         padding:10px;border:1px solid #999;border-radius:4px;font-size:13px;">
      <b>Stop QC</b><br>
      <span style="color:#d7191c;">●</span> Voluntary stop centroid<br>
      <span style="color:#fdae61;">●</span> Involuntary stop centroid<br>
      <span style="color:#2c7bb6;">●</span> Driving points<br>
      <span style="color:#fdae61;">●</span> Involuntary points<br>
      <span style="color:#d7191c;">●</span> Voluntary points<br>
      Gray line = trajectory context
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))

    for i, (_idx, event) in enumerate(sample.iterrows(), start=1):
        track = _context_track(points, event, pad_s=pad_s)
        _add_event_layer(fmap, event, track, idx=i)

    # Fit bounds to sampled centroids
    lats = sample["centroid_lat"].tolist()
    lons = sample["centroid_lon"].tolist()
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return fmap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    stops_path = args.output_dir / "stop_events.parquet"
    points_path = args.output_dir / "decoded_points.parquet"
    if not stops_path.exists() or not points_path.exists():
        raise SystemExit(f"Missing parquet outputs under {args.output_dir}")

    stops = pd.read_parquet(stops_path)
    points = pd.read_parquet(points_path)
    points["timestamp"] = pd.to_datetime(points["timestamp"])

    filtered = _filter_stops(stops, args)
    n = min(int(args.n), len(filtered))
    sample = filtered.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)

    fmap = build_qc_map(stops, points, sample, pad_s=float(args.pad_s))
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(args.html_out))

    # Also write a sidecar table for the sampled events
    meta_path = args.html_out.with_suffix(".json")
    meta = sample[
        [
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
    ].copy()
    meta["start_time"] = meta["start_time"].astype(str)
    meta["end_time"] = meta["end_time"].astype(str)
    meta_path.write_text(json.dumps(meta.to_dict(orient="records"), indent=2))

    print(f"Sampled {n} / {len(filtered)} matching stops (seed={args.seed})")
    print(f"Wrote map:  {args.html_out.resolve()}")
    print(f"Wrote meta: {meta_path.resolve()}")
    print("Open the HTML file in a browser and click red markers to inspect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
