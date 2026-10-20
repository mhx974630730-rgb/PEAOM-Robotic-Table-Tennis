#!/usr/bin/env python3
"""Shared experiment configuration for simulation and physical evaluation.

Edit only the USER CONFIGURATION section below, save the file, and then
launch the ROS/Python nodes normally. All processes import the same values;
terminal environment variables are not required or read by this module.
"""

import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


# ============================================================================
# USER CONFIGURATION -- normally edit only this section
# ============================================================================

# "sim_train": PyBullet training; launch sim_env.py, controller_node.py,
#              and train.py --new.
# "real_eval": zero-shot physical evaluation; launch the camera, robot,
#              controller_node.py, and outcome_recorder.py. Do NOT run train.py.
USER_STAGE = "sim_train"

# Exactly one observation method. During real_eval this selects the model
# trained with that method; it does not alter the physical camera trajectory.
# Choices: ideal, gaussian_residual, gaussian_raw, peaom, peaom_residual
USER_OBSERVATION_MODE = "gaussian_residual"

# Policy action interpretation. Formal paper experiments should use "k".
# "t_delay" remains available for debugging/compatibility.
USER_TIMING_MODE = "k"

# Formal seeds: change this to 1, 2, 3, 4, or 5.
USER_RANDOM_SEED = 1

# Leave blank to generate a stable ID such as train_peaom_seed1 or
# real_peaom_seed1. Set a new explicit value when intentionally repeating a
# run, so that data are not appended to an earlier run directory.
USER_RUN_ID = ""

# Output group name below models/, train_data or physical_data/, and
# experiment_logs/. Normally keep this as "formal".
USER_OUTPUT_GROUP = "formal"

# Leave these paths blank to use the automatically generated paths. A custom
# relative path is resolved from the directory containing this file (for
# example, sim_and_real2 or sim_and_real3).
USER_MODEL_DIR = ""
USER_DATA_DIR = ""
USER_LOG_ROOT = ""
USER_LINEAR_CORRECTION_PATH = ""  # blank -> hardware_sys/linear_correction.npz

# Observation-model parameters in metres.
USER_RESIDUAL_NOISE_STD = [0.0055, 0.0040, 0.0038]
USER_RAW_NOISE_STD = [0.0425, 0.0267, 0.0508]

# 0 means all valid samples before the prediction plane; otherwise use at
# least 4 samples.
USER_FIT_WINDOW_SIZE = 0

# Reviewer 2 Q2 trajectory collection. When enabled, sim_env.py stores one
# dedicated incoming-trajectory file in the current RUN_DIR. A trajectory is
# committed only after the true ball position first crosses the prediction
# plane in the +x direction, so post-hit and incomplete trajectories are not
# included. This switch does not change the observations published to ROS.
USER_ENABLE_Q2_TRAJECTORY_LOG = False
USER_Q2_PREDICT_PLANE_X = -0.3

# Used only to warn when a completed trajectory contains too few samples for
# the largest fitting window planned in the sensitivity analysis.
USER_Q2_REQUIRED_SAMPLES = 17

# Formal experiment switches. Keep all four False for reported experiments.
USER_DEBUG_AIM_AT_PREDICTED_BALL = False
USER_DEBUG_FIXED_POSE = False
USER_DEBUG_FIXED_VX = False
USER_ENABLE_SAFE_HIT_RANGE = False

# Zero-shot protection. Keep False. Manual outcome recording is enabled
# automatically in real_eval and disabled in sim_train.
USER_ALLOW_REAL_FINETUNE = False


# ============================================================================
# RESOLVED CONFIGURATION -- do not normally edit below this line
# ============================================================================

_STAGES = {"sim_train", "real_eval"}
_OBSERVATION_MODES = {
    "ideal", "gaussian_residual", "gaussian_raw", "peaom", "peaom_residual"
}
_TIMING_MODES = {"k", "t_delay"}


def _checked_choice(name, value, allowed):
    resolved = str(value).strip().lower()
    if resolved not in allowed:
        raise ValueError(f"{name}={value!r}; expected one of {sorted(allowed)}")
    return resolved


def _resolve_path(value, default):
    text = str(value).strip()
    path = Path(text).expanduser() if text else Path(default)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


EXPERIMENT_STAGE = _checked_choice("USER_STAGE", USER_STAGE, _STAGES)
OBSERVATION_MODE = _checked_choice(
    "USER_OBSERVATION_MODE", USER_OBSERVATION_MODE, _OBSERVATION_MODES
)
TIMING_MODE = _checked_choice(
    "USER_TIMING_MODE", USER_TIMING_MODE, _TIMING_MODES
)

RANDOM_SEED = int(USER_RANDOM_SEED)
if RANDOM_SEED < 0:
    raise ValueError("USER_RANDOM_SEED must be a non-negative integer")

FIT_WINDOW_SIZE = int(USER_FIT_WINDOW_SIZE)
if FIT_WINDOW_SIZE != 0 and FIT_WINDOW_SIZE < 4:
    raise ValueError("USER_FIT_WINDOW_SIZE must be 0 (all) or at least 4")

ENABLE_Q2_TRAJECTORY_LOG = bool(USER_ENABLE_Q2_TRAJECTORY_LOG)
Q2_PREDICT_PLANE_X = float(USER_Q2_PREDICT_PLANE_X)
if not np.isfinite(Q2_PREDICT_PLANE_X):
    raise ValueError("USER_Q2_PREDICT_PLANE_X must be finite")

Q2_REQUIRED_SAMPLES = int(USER_Q2_REQUIRED_SAMPLES)
if Q2_REQUIRED_SAMPLES < 4:
    raise ValueError("USER_Q2_REQUIRED_SAMPLES must be at least 4")

output_group = str(USER_OUTPUT_GROUP).strip()
if not output_group or output_group in {".", ".."} or "/" in output_group or "\\" in output_group:
    raise ValueError("USER_OUTPUT_GROUP must be one simple directory name")

METHOD_NAME = OBSERVATION_MODE
IS_REAL_ROBOT = EXPERIMENT_STAGE == "real_eval"
TRAIN_MODE = "real" if IS_REAL_ROBOT else "sim"
ALLOW_REAL_FINETUNE = bool(USER_ALLOW_REAL_FINETUNE)
REQUIRE_MANUAL_OUTCOME = IS_REAL_ROBOT

DEBUG_AIM_AT_PREDICTED_BALL = bool(USER_DEBUG_AIM_AT_PREDICTED_BALL)
DEBUG_FIXED_POSE = bool(USER_DEBUG_FIXED_POSE)
DEBUG_FIXED_VX = bool(USER_DEBUG_FIXED_VX)
ENABLE_SAFE_HIT_RANGE = bool(USER_ENABLE_SAFE_HIT_RANGE)

K_RANGE = (0.001, 0.01)
T_DELAY_RANGE = (-0.05, 0.05)
RESIDUAL_NOISE_STD = np.asarray(USER_RESIDUAL_NOISE_STD, dtype=np.float64)
RAW_NOISE_STD = np.asarray(USER_RAW_NOISE_STD, dtype=np.float64)
if RESIDUAL_NOISE_STD.shape != (3,) or RAW_NOISE_STD.shape != (3,):
    raise ValueError("Noise standard deviations must each contain exactly 3 values")
if np.any(RESIDUAL_NOISE_STD < 0.0) or np.any(RAW_NOISE_STD < 0.0):
    raise ValueError("Noise standard deviations cannot be negative")

seed_dir = f"seed{RANDOM_SEED}"
default_model_dir = BASE_DIR / "models" / output_group / METHOD_NAME / seed_dir
data_root_name = "physical_data" if IS_REAL_ROBOT else "train_data"
default_data_dir = BASE_DIR / data_root_name / output_group / METHOD_NAME / seed_dir
default_log_root = BASE_DIR / "experiment_logs" / output_group

MODEL_DIR = str(_resolve_path(USER_MODEL_DIR, default_model_dir))
DATA_DIR = str(_resolve_path(USER_DATA_DIR, default_data_dir))
LOG_ROOT = _resolve_path(USER_LOG_ROOT, default_log_root)
LINEAR_CORRECTION_PATH = _resolve_path(
    USER_LINEAR_CORRECTION_PATH, PROJECT_DIR / "linear_correction.npz"
)

run_prefix = "real" if IS_REAL_ROBOT else "train"
automatic_run_id = f"{run_prefix}_{METHOD_NAME}_seed{RANDOM_SEED}"
RUN_ID = str(USER_RUN_ID).strip() or automatic_run_id
if RUN_ID in {".", ".."} or "/" in RUN_ID or "\\" in RUN_ID:
    raise ValueError("USER_RUN_ID must be one simple directory name")
RUN_DIR = LOG_ROOT / RUN_ID


def ensure_run_dir():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR


def metadata():
    return {
        "run_id": RUN_ID,
        "experiment_stage": EXPERIMENT_STAGE,
        "method": METHOD_NAME,
        "observation_mode": OBSERVATION_MODE,
        "model_dir": MODEL_DIR,
        "data_dir": DATA_DIR,
        "k_range": list(K_RANGE),
        "t_delay_range": list(T_DELAY_RANGE),
        "residual_noise_std_m": RESIDUAL_NOISE_STD.tolist(),
        "raw_noise_std_m": RAW_NOISE_STD.tolist(),
        "linear_correction_path": str(LINEAR_CORRECTION_PATH),
        "debug_aim": DEBUG_AIM_AT_PREDICTED_BALL,
        "debug_pose": DEBUG_FIXED_POSE,
        "debug_vx": DEBUG_FIXED_VX,
        "safe_clip": ENABLE_SAFE_HIT_RANGE,
        "is_real_robot": IS_REAL_ROBOT,
        "allow_real_finetune": ALLOW_REAL_FINETUNE,
        "require_manual_outcome": REQUIRE_MANUAL_OUTCOME,
        "random_seed": RANDOM_SEED,
        "fit_window_size": FIT_WINDOW_SIZE,
        "q2_trajectory_log_enabled": ENABLE_Q2_TRAJECTORY_LOG,
        "q2_predict_plane_x_m": Q2_PREDICT_PLANE_X,
        "q2_required_samples": Q2_REQUIRED_SAMPLES,
    }


def write_metadata():
    run_dir = ensure_run_dir()
    path = run_dir / "run_metadata.json"
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata(), f, ensure_ascii=False, indent=2)
    return path
