# PEAOM for Robotic Table Tennis
Official repository: https://github.com/mhx974630730-rgb/PEAOM-Robotic-Table-Tennis
Research code for **“Perception-Error-Aware Observation Modeling for Zero-Shot Sim-to-Real Control of Robotic Table Tennis.”**

This project was developed by Hongxu Ma, Jianyin Fan, Haoran Xu, and Qiang Wang in the **Department of Control Science and Engineering, School of Astronautics, Harbin Institute of Technology**, Harbin 150001, China.

- Corresponding author: Qiang Wang (`wangqiang@hit.edu.cn`)
- Technical contact: Hongxu Ma (`19b904043@stu.hit.edu.cn`)
- Institution: [Harbin Institute of Technology](https://en.hit.edu.cn/)

## Scope of this release

The repository contains the code and compact reference artifacts required to reproduce the simulation workflow and inspect the observation-modeling pipeline:

- PyBullet robotic table-tennis environment;
- SAC-style policy training and inference;
- ideal, residual-Gaussian, raw-Gaussian, PEAOM, and PEAOM-with-residual observation modes;
- the calibrated affine perception model used by PEAOM;
- timing-parameter conversion and tests;
- URDF and mesh assets used by the simulator;
- the custom ROS message required by the simulation nodes;
- one reference actor checkpoint for each main comparison method;
- optional vision, calibration, trajectory-collection, and analysis utilities.

The following are intentionally excluded:

- manufacturer-provided robot communication source code;
- the physical robot actuation bridge;
- the vendor camera SDK, its Python binding, and device-specific camera configuration files;
- raw checkerboard images, generated trajectories, replay data, logs, critic checkpoints, and intermediate training files.

## Repository layout

| Path | Purpose |
| --- | --- |
| `sim_and_real3/` | Simulation, controller, training, timing conversion, logging, tests, and analysis. |
| `sim_and_real3/models/formal/` | Reference actor checkpoints for `ideal`, `gaussian_residual`, and `peaom`. |
| `urdf_folder/` | PyBullet robot, ball, table, net, and mesh assets. |
| `ros/mhx_msg_pkg/` | ROS 1 definition of `HitCommand`. |
| `camera_pram_dec/` | Optional binocular localization and camera-calibration utilities. |
| `pram_jz/` | Optional paired-position collection and affine perception-model fitting. |
| `sys_pram_dec/` | Optional physical trajectory collection and dynamics identification. |
| `linear_correction.npz` | Affine PEAOM mapping stored as arrays `A` and `b`. |
| `CITATION.cff` | Software and manuscript citation metadata. |
| `THIRD_PARTY_NOTICES.md` | Components that must be obtained separately. |

## Requirements

The complete simulation workflow was designed for Linux with ROS 1. ROS Noetic with Python 3.8 is recommended.

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Required ROS packages are `rospy`, `std_msgs`, `geometry_msgs`, `message_generation`, and `message_runtime`.

The optional physical-vision tools additionally require a vendor-installed MindVision-compatible camera runtime and an importable `mvsdk` Python binding. Those vendor files are not redistributed here.

## Build the ROS message

```bash
mkdir -p ~/catkin_ws/src
cp -r ros/mhx_msg_pkg ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Configure an experiment

Edit the user section at the top of `sim_and_real3/experiment_config.py`. No terminal environment variables are required.

The main fields are:

- `USER_STAGE`: use `sim_train` for simulation;
- `USER_OBSERVATION_MODE`: `ideal`, `gaussian_residual`, `gaussian_raw`, `peaom`, or `peaom_residual`;
- `USER_TIMING_MODE`: use `k` for the paper experiments; `t_delay` is retained as an equivalent compatibility mode;
- `USER_RANDOM_SEED`: random seed;
- `USER_RUN_ID`: leave empty for the automatically generated run identifier;
- `USER_ENABLE_Q2_TRAJECTORY_LOG`: enable only when collecting state-estimation sensitivity trajectories;
- `USER_LINEAR_CORRECTION_PATH`: leave empty to use `linear_correction.npz` in the repository root.

Keep the debug and safety-override switches set to `False` when reproducing the reported experiments.

## Train in simulation

From the repository root, launch the following processes in separate terminals:

```bash
roscore
```

```bash
python3 sim_and_real3/sim_env.py
```

```bash
python3 sim_and_real3/controller_node.py
```

```bash
python3 sim_and_real3/train.py --new
```

Use `--new` only to start a new run. Omit it to resume a run whose actor, critics, and training counter are already present in the model directory resolved by `experiment_config.py`.

Generated replay samples, logs, losses, critics, and counters are ignored by Git.

## Evaluate a reference actor

Reference actor checkpoints are provided for the three principal comparison methods:

```text
sim_and_real3/models/formal/ideal/seed1/sim_actor_weights.pth
sim_and_real3/models/formal/gaussian_residual/seed1/sim_actor_weights.pth
sim_and_real3/models/formal/peaom/seed1/sim_actor_weights.pth
```

Set `USER_STAGE = "sim_train"`, select the matching observation mode and seed, then start `sim_env.py` and `controller_node.py`. Do not start `train.py` for evaluation-only runs.

The `gaussian_raw` mode is retained for the raw-RMSE noise ablation, but no trained checkpoint is supplied because this condition did not produce a usable policy in the reported high-noise setting.

## Observation models

The actor emits seven normalized values in `[-1, 1]`. They are mapped to target racket position, orientation offsets, timing, and racket velocity.

The PEAOM observation is generated by applying

```text
p_observed = A @ p_true + b
```

to every simulated ball-trajectory sample before the same state-fitting procedure used by the physical controller. The residual modes add calibrated zero-mean Gaussian residuals after this deterministic mapping.

Positions are expressed in metres, linear velocities in metres per second, angles in radians, and ROS timestamps in seconds.

## Calibration and model fitting

The compact runtime camera parameters are stored in `camera_pram_dec/camera_pram/`. Raw checkerboard images are not included; users should capture images for their own camera pair and workspace.

Relevant utilities are:

```bash
python3 camera_pram_dec/get_pic.py
python3 camera_pram_dec/pram_w.py
python3 camera_pram_dec/pram_wc.py
python3 pram_jz/camera_info_get2.py
python3 pram_jz/camera_sim_to_real.py \
  --input pram_jz/data/calibration_data_accumulated_swapped.npz \
  --output pram_jz/data/linear_correction.npz
```

Validate coordinate order, units, residuals, and workspace coverage before replacing the root `linear_correction.npz`.

## Tests and analysis

The timing conversion tests do not require ROS:

```bash
python3 sim_and_real3/test_timing_mapping.py
```

The analysis utilities expose command-line help:

```bash
python3 sim_and_real3/analyze_experiment_logs.py --help
python3 sim_and_real3/analyze_state_sensitivity.py --help
```

`sys_pram_dec/sys_pram_tra.py` collects physical ball trajectories and `sys_pram_dec/sys_pram_cal.py` estimates dynamics parameters from them. Generated trajectory files are intentionally excluded.

## Physical deployment boundary

The repository does not include the robot execution program. `controller_node.py` publishes `mhx_msg_pkg/HitCommand`; a separately authorized robot bridge must subscribe to that message and implement the physical command without changing its semantics.

Physical robot operation can be hazardous. Validate the full message path in simulation, apply conservative limits, provide an emergency stop, and keep people outside the robot workspace.

## Citation

Citation metadata are provided in `CITATION.cff`. Update the preferred citation with the final journal DOI, volume, issue, and pages after publication.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` and `NOTICE` for details. The license applies only to the material distributed in this repository; excluded manufacturer and camera-vendor components remain subject to their respective owners' terms.
