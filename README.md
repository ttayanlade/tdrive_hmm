# T-Drive 3-State Gaussian HMM

Production pipeline for trajectory segmentation and stop classification on the [Microsoft T-Drive](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) dataset.

**States:** `Driving` · `Possible Involuntary Stop` · `Possible Voluntary Stop`

## Pipeline

1. **Preprocess** — Beijing bbox filter, teleport removal (>45 m/s), UTM 50N projection, cross-gap voluntary-stop bridging (`δt > 30s` & `d ≤ 35m`), high-frequency segment retention (median `δt ≤ 15s`, ≥20 points, span ≥60s).
2. **Smooth** — 2D constant-velocity Kalman filter + RTS smoother (`σ_GPS=10m`, `σ_a=1.5 m/s²`).
3. **Features** — `O_t = [log(v+0.05), a, log(R+0.1)]` with 15s jitter radius; `StandardScaler` fit on train taxis only.
4. **HMM** — Constrained 3-state `GaussianHMM` (`A₁₂=A₂₁=0`, `params="stmc"`); semantic remap by mean speed + self-transition persistence.
5. **Post-process** — Viterbi runs, bridged durations, quality flags; export parquet + diagnostics.

Taxi-level 80/20 split (`seed=42`) prevents scaler / HMM leakage across trajectories.


## Run

```bash
# Full corpus (~10,357 taxis)
python -m tdrive_hmm \
  --data-dir /path/to/taxi_log_2008_by_id \
  --output-dir outputs \
  --n-jobs -1
```

Local data path in this workspace:

`../trajectory_data/release/taxi_log_2008_by_id`

Note: only a few percent of taxis retain high-frequency segments after the
median-`δt ≤ 15s` / ≥20-point filters (most polling is ~60–300s standby). Use the
full corpus for stable HMM estimates.
