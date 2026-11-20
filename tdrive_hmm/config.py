"""Pipeline constants for the T-Drive 3-state Gaussian HMM."""

from __future__ import annotations

# Geographic / projection
BEIJING_LAT_MIN = 39.4
BEIJING_LAT_MAX = 41.1
BEIJING_LON_MIN = 115.4
BEIJING_LON_MAX = 117.5
UTM_EPSG = "EPSG:32650"  # UTM Zone 50N

# Teleport / gap / segment filters
MAX_RAW_SPEED_M_S = 45.0
GAP_THRESHOLD_S = 30.0
BRIDGE_MAX_DISPLACEMENT_M = 35.0
HIGH_FREQ_MEDIAN_DT_S = 15.0
MIN_SEGMENT_POINTS = 20
MIN_SEGMENT_SPAN_S = 60.0

# Kalman / RTS
GPS_SIGMA_M = 10.0
PROCESS_ACCEL_SIGMA = 1.5  # m/s^2
ACCEL_CLIP_MIN = -6.0
ACCEL_CLIP_MAX = 6.0

# Features
JITTER_WINDOW_S = 15.0
SPEED_LOG_EPS = 0.05
JITTER_LOG_EPS = 0.1

# HMM
N_STATES = 3
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 100
HMM_TOL = 1e-4
HMM_RANDOM_STATE = 42

# Semantic labels (after remapping)
STATE_DRIVING = "Driving"
STATE_INVOLUNTARY = "Possible Involuntary Stop"
STATE_VOLUNTARY = "Possible Voluntary Stop"

STATE_LABELS = {
    0: STATE_DRIVING,
    1: STATE_INVOLUNTARY,
    2: STATE_VOLUNTARY,
}

# Train / test
TRAIN_FRACTION = 0.80
RANDOM_SEED = 42

# I/O column names
RAW_COLUMNS = ["taxi_id", "timestamp", "longitude", "latitude"]
