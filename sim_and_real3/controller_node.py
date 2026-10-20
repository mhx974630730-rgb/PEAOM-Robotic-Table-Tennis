#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import threading
import queue
import serial
from std_msgs.msg import String, Bool, Int32
from geometry_msgs.msg import PointStamped
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from scipy.optimize import least_squares
import os
import torch
from network import Actor  # 复用训练时的 Actor 网络
import csv
import json
import hashlib

# 自定义消息（需提前创建 msg/HitCommand.msg 并编译）
from mhx_msg_pkg.msg import HitCommand   # 替换 your_package 为实际包名
from experiment_config import (
    TIMING_MODE, K_RANGE, METHOD_NAME, OBSERVATION_MODE,
    RUN_ID, RUN_DIR, IS_REAL_ROBOT, DEBUG_AIM_AT_PREDICTED_BALL,
    DEBUG_FIXED_POSE, DEBUG_FIXED_VX, ENABLE_SAFE_HIT_RANGE,
    FIT_WINDOW_SIZE, REQUIRE_MANUAL_OUTCOME, RANDOM_SEED,
    ensure_run_dir, write_metadata,
)
from config import SIM_WEIGHT_PATH, DATA_DIR
from timing_mapping import (
    arrival_time_from_k as _arrival_time_from_k,
    k_from_arrival_time as _k_from_arrival_time,
    resolve_policy_timing,
)


# ==========================================
# 1. 配置与常量
# ==========================================
@dataclass
class Config:
    '''
    state 结构: [hit_x, hit_y, hit_z, vx, vy, vz, target_x, target_y, target_v]
    action 结构：[px, py, pz, rx, ry, timing, vx_cmd]
    timing 由 experiment_config.py 中的 USER_TIMING_MODE 解释为 k 或 t_delay
    reward = [dy, dz, rt, rh, px, py, dv]
    '''
    # --- 强化学习模型参数 --
    state_dim = 9
    action_dim = 7
    #
    model_path: str = SIM_WEIGHT_PATH
    # --- 物理参数 ---
    table_z: float = 0.0
    net_x: float = 0.0
    net_height: float = 0.0891
    ball_radius: float = 0.02
    hit_plane_x: float = 0.8
    gravity: float = 9.81

    # --- 弹跳模型 (简化恢复系数) ---
    restitution_z: float = 0.85
    friction_xy: float = 0.95

    # --- 系统参数 ---
    control_freq: int = 30
    tracking_timeout: float = 1.0
    is_real_robot: bool = IS_REAL_ROBOT
    serial_port: str = '/dev/ttyUSB0'
    serial_baud: int = 9600

    # --- 模式切换 ---
    prediction_mode: str = 'INIT_PLANE'  # 'HIT_PLANE'/'INIT_PLANE'
    enable_success_stats: bool = True

cfg = Config()
DEBUG_FIXED_VX_VALUE = 1.2           # 先用 1.0~1.2 测试
# ==========================================
# Debug RZ adjustment according to hitting y
# ==========================================
DEBUG_LINEAR_RZ_BY_Y = True

# y 达到该范围边界时，rz 达到最大变化量
DEBUG_RZ_Y_RANGE = 0.3

# rz 最大变化量，单位 rad
DEBUG_RZ_MAX_DELTA = np.pi / 8.0

SAFE_HIT_X_MIN = 0.78
SAFE_HIT_X_MAX = 0.80

SAFE_HIT_Y_MIN = -0.3
SAFE_HIT_Y_MAX = 0.3

SAFE_HIT_Z_MIN = 0.10
SAFE_HIT_Z_MAX = 0.40

SAFE_VX_MIN = 1.0
SAFE_VX_MAX = 4.0
# ==========================================
# 2. 数学与物理工具库
# ==========================================

class PhysicsEngine:
    @staticmethod
    def arrival_time_from_k(x0: float, vx0: float, x_hit: float, k: float) -> Optional[float]:
        """Paper timing model, implemented in the ROS-independent helper."""
        return _arrival_time_from_k(x0, vx0, x_hit, k)

    @staticmethod
    def k_from_arrival_time(x0: float, vx0: float, x_hit: float,
                            arrival_time: float) -> Optional[float]:
        """Invert the paper timing equation and return the equivalent k."""
        return _k_from_arrival_time(x0, vx0, x_hit, arrival_time)

    @staticmethod
    def fit_parabola(points: np.ndarray, times: np.ndarray):
        if len(points) < 4: return None
        t0 = times[0]
        t_rel = times - t0
        A = np.vstack([t_rel**2, t_rel, np.ones_like(t_rel)]).T
        coeffs = []
        for i in range(3):
            try:
                c, _, _, _ = np.linalg.lstsq(A, points[:, i], rcond=None)
                coeffs.extend(c)
            except: return None
        return {'coeffs': np.array(coeffs), 't0': t0}

    @staticmethod
    def get_state(coeffs_dict: dict, t: float) -> Tuple[np.ndarray, np.ndarray]:
        coeffs = coeffs_dict['coeffs']
        t0 = coeffs_dict['t0']
        t_rel = t - t0
        pos = np.array([
            coeffs[0]*t_rel**2 + coeffs[1]*t_rel + coeffs[2],
            coeffs[3]*t_rel**2 + coeffs[4]*t_rel + coeffs[5],
            coeffs[6]*t_rel**2 + coeffs[7]*t_rel + coeffs[8]
        ])
        vel = np.array([
            2*coeffs[0]*t_rel + coeffs[1],
            2*coeffs[3]*t_rel + coeffs[4],
            2*coeffs[6]*t_rel + coeffs[7]
        ])
        return pos, vel

    @staticmethod
    def solve_time_for_axis(coeffs_dict: dict, axis: int, target_val: float, t_min_global: float) -> Optional[float]:
        coeffs = coeffs_dict['coeffs']
        t0 = coeffs_dict['t0']
        a, b, c = coeffs[axis*3], coeffs[axis*3+1], coeffs[axis*3+2]
        A_q, B_q, C_q = a, b, c - target_val

        if abs(A_q) < 1e-9:
            if abs(B_q) < 1e-9: return None
            t_rel = -C_q / B_q
            return t0 + t_rel if (t0 + t_rel) > t_min_global else None

        delta = B_q**2 - 4*A_q*C_q
        if delta < 0: return None
        sqrt_d = np.sqrt(delta)
        t_rels = [(-B_q + sqrt_d)/(2*A_q), (-B_q - sqrt_d)/(2*A_q)]
        candidates = [t0 + tr for tr in t_rels if (t0 + tr) > t_min_global]
        return min(candidates) if candidates else None

    @staticmethod
    def find_landing_time(coeffs_dict: dict, t_min_global: float) -> Optional[float]:
        coeffs = coeffs_dict['coeffs']
        t0 = coeffs_dict['t0']
        a, b, c = coeffs[6], coeffs[7], coeffs[8]
        A_q, B_q, C_q = a, b, c - cfg.table_z
        if abs(A_q) < 1e-9: return None
        delta = B_q**2 - 4*A_q*C_q
        if delta < 0: return None
        sqrt_d = np.sqrt(delta)
        t_rels = [(-B_q + sqrt_d)/(2*A_q), (-B_q - sqrt_d)/(2*A_q)]
        for tr in t_rels:
            tg = t0 + tr
            if tg > t_min_global:
                vz = 2*a*tr + b
                if vz < 0: return tg
        return None

    @staticmethod
    def simulate_bounce(pos_in: np.ndarray, vel_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vel_out = vel_in.copy()
        vel_out[2] = -vel_in[2] * cfg.restitution_z
        vel_out[:2] *= cfg.friction_xy
        pos_out = pos_in.copy()
        pos_out[2] = cfg.table_z + 0.001
        return pos_out, vel_out

# ==========================================
# 3. RL 大脑接口
# ==========================================
class RL_Brain:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = Actor().to(self.device)
        self.model.eval()
        self.load_model()

        if cfg.is_real_robot:
            # 实物基准姿态：欧拉角（弧度）  (-89.246°, 3.557°, 92.869°)
            deg_to_rad = np.pi / 180.0
            self.default_pose = {
                'rx': -78.22 * deg_to_rad,
                'ry': -8.942 * deg_to_rad,
                'rz': 89.77 * deg_to_rad
            }
        else:
            self.default_pose = {
                'rx': np.pi / 2,
                'ry': -0.7697,
                'rz': 1.44221
            }
        if cfg.prediction_mode == 'HIT_PLANE':
            self.action_ranges = {
                'px': (-0.1, 0.1),
                'py': (-0.1, 0.1),
                'pz': (0.0, 0.2),
                'rx': (-np.pi/4, np.pi/4),
                'ry': (-np.pi/4, np.pi/4),
                # In t_delay mode this remains a normalized selector.  Its
                # physical delay is resolved per ball from K_RANGE later.
                'timing': K_RANGE if TIMING_MODE == 'k' else (-1.0, 1.0),
                'vx_cmd': (0.0, 4.0)
            }
        elif cfg.prediction_mode == 'INIT_PLANE':
            self.action_ranges = {
                'px': (0.79, 0.80),
                'py': (-0.35, 0.35),
                'pz': (0.1, 0.4),
                'rx': (-np.pi/4, np.pi/4),
                'ry': (-np.pi/4, np.pi/4),
                # In t_delay mode this remains a normalized selector.  Its
                # physical delay is resolved per ball from K_RANGE later.
                'timing': K_RANGE if TIMING_MODE == 'k' else (-1.0, 1.0),
                'vx_cmd': (0.0, 4.0)
            }

    def load_model(self):
        try:
            checkpoint = torch.load(cfg.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"[Brain] Weights loaded from {cfg.model_path}")
        except Exception as e:
            print(f"[Brain] Load failed: {e}")
            if cfg.is_real_robot:
                raise RuntimeError(
                    f"Refusing to run the physical robot without a valid policy: {cfg.model_path}"
                ) from e

    def get_action(self, state: np.ndarray, deterministic=False) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor, _ = self.model(state_tensor, deterministic=deterministic, return_log_prob=False)
            raw_action = action_tensor.cpu().numpy()[0]

        mapped_action = []
        for i, (low, high) in enumerate(self.action_ranges.values()):
            mapped = low + (raw_action[i] + 1.0) * (high - low) / 2.0
            mapped_action.append(mapped)
        return np.array(raw_action), np.array(mapped_action)

# ==========================================
# 3. 核心控制器类
# ==========================================
class PingPongAgent:
    def __init__(self, max_episodes = 500):
        # Make policy sampling and randomized return targets reproducible for
        # the five formal seeds.  Each ROS process receives the same seed via
        # USER_RANDOM_SEED in experiment_config.py.
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
        self.idle_timeout_timer = rospy.Timer(rospy.Duration(2.0), self.check_idle_timeout)
        self.lock = threading.RLock()
        self.data_queue = queue.Queue(maxsize=200)
        self.time_advance = 0.06
        self.epoise = 0
        self.phase = "IDLE"
        self.traj_buffer: List[Dict] = []
        self.last_msg_time = rospy.Time.now().to_sec()
        self.max_traj_length = 200
        self.t_plane_crossing = None
        self.idx_bounce1 = None
        self.idx_hit = None
        self.round_start_time = None
        if cfg.is_real_robot:
            self.round_timeout_duration = 10.0
        else:
            self.round_timeout_duration = 3.0
        self.predict_plane_x = -0.3
        self.fit_net = None
        self.fit_pre_hit = None
        self.fit_point_count = 0
        self.last_prediction_issue = ''
        self.max_episodes = max_episodes

        self.current_state_vec = None
        self.current_action_vec = None
        self.true_current_action_vec = None
        self.predicted_hit_time = None

        self.brain = RL_Brain()
        self.serial_port = None
        if cfg.is_real_robot:
            try:
                self.serial_port = serial.Serial(cfg.serial_port, cfg.serial_baud, timeout=1)
                rospy.loginfo(f"Serial port opened successfully")
            except Exception as e:
                rospy.loginfo(f"Serial port open failed: {e}")
                self.serial_port = None

        self.reset_round()
        self.sub = rospy.Subscriber('/ball_position', PointStamped, self.callback_ball, queue_size=5)
        self.hit_cmd_pub = rospy.Publisher('/controller_node/action_publisher', HitCommand, queue_size=1)
        self.pub_sim_start = rospy.Publisher('/sim/game_start', Bool, queue_size=1)
        self.episode_pub = rospy.Publisher('/experiment/action_episode', Int32, queue_size=1, latch=True)
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(DATA_DIR, "buffer_info.npz")
        ensure_run_dir()
        write_metadata()
        self.action_log_path = os.path.join(str(RUN_DIR), "actions.csv")
        self.trajectory_log_path = os.path.join(str(RUN_DIR), "trajectory_points.csv")
        self.auto_result_log_path = os.path.join(str(RUN_DIR), "auto_results.csv")
        self.manual_outcome_log_path = os.path.join(str(RUN_DIR), "manual_outcomes.csv")
        self.prediction_failure_log_path = os.path.join(
            str(RUN_DIR), "prediction_failures.csv"
        )
        self.logged_result_episodes = set()
        self.logged_failure_episodes = set()
        self.last_action_episode = None
        self.last_timing_value = None
        self.last_t_delay = None
        self.last_k = None
        self.awaiting_outcome = False
        self.pending_outcome_episode = None
        self._init_experiment_logs()
        self.logged_result_episodes = self._read_logged_episode_ids(
            self.auto_result_log_path
        )
        self.logged_failure_episodes = self._read_logged_episode_ids(
            self.prediction_failure_log_path
        )
        self._write_controller_metadata()
        self.outcome_sub = rospy.Subscriber(
            '/experiment/trial_outcome', String, self.callback_manual_outcome, queue_size=10
        )
        self.send_times = []
        self.consecutive_diffs = []
        #log
        self.reward_log_path = os.path.join(self.current_dir, "reward_log.csv")
        if not os.path.exists(self.reward_log_path):
            with open(self.reward_log_path, 'w') as f:
                f.write("episode,dy,dz,dt,h,x,y,v\n")
        self.total_rounds = 0      # 总回合数
        self.success_hits = 0      # 成功回击到对侧球台的次数
        self.stats_interval = 50   # 每多少个回合打印一次成功率
        self.test_flag = False
        self.current_epoise = 0
        self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.process_thread.start()

    @staticmethod
    def _ensure_csv(path, header):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'r', newline='') as f:
                existing_header = next(csv.reader(f), [])
            if existing_header != list(header):
                raise RuntimeError(
                    f"Log schema mismatch in {path}. Set a new USER_RUN_ID in "
                    "experiment_config.py instead of appending to an older log."
                )
            return
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)

    @staticmethod
    def _read_logged_episode_ids(path):
        """Return episode IDs already present in a structured CSV log."""
        episode_ids = set()
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return episode_ids
        with open(path, 'r', newline='') as f:
            for row in csv.DictReader(f):
                try:
                    episode_ids.add(int(row.get('episode', '')))
                except (TypeError, ValueError):
                    continue
        return episode_ids

    def _init_experiment_logs(self):
        state_cols = [f'state_{x}' for x in ('x0','y0','z0','vx0','vy0','vz0','target_x','target_y','target_v')]
        action_names = ('px','py','pz','rx','ry','timing','vx')
        raw_cols = [f'raw_{x}' for x in action_names]
        mapped_cols = [f'mapped_{x}' for x in action_names]
        self.action_log_header = [
            'run_id','method','observation_mode','episode','ros_time',
            *state_cols, *raw_cols, *mapped_cols,
            'pred_hit_x','pred_hit_y','pred_hit_z','pred_vx','pred_vy','pred_vz',
            'policy_target_x','policy_target_y','policy_target_z',
            'executed_target_x','executed_target_y','executed_target_z',
            'target_clipped','timing_value','k','t_delay','predicted_hit_time',
            'scheduled_hit_rel','time_to_hit','cmd_vx','debug_aim','debug_pose','debug_vx',
            'safe_clip','fit_point_count','command_published','rejection_reason'
        ]
        self._ensure_csv(self.action_log_path, self.action_log_header)
        self._ensure_csv(self.trajectory_log_path, [
            'run_id','method','observation_mode','episode','t_rel','x','y','z','vx','vy','vz','phase'
        ])
        self._ensure_csv(self.auto_result_log_path, [
            'run_id','method','episode','trajectory_points','reward_valid','hit_success',
            'dy','dz','dt','net_height_reward','landing_x_reward','landing_y_reward','velocity_reward'
        ])
        self._ensure_csv(self.manual_outcome_log_path, [
            'run_id','method','recorded_at','episode','outcome','note'
        ])
        self._ensure_csv(self.prediction_failure_log_path, [
            'run_id','method','observation_mode','episode','ros_time','phase',
            'failure_reason','trajectory_points','fit_point_count',
            'state_available','action_available','command_published'
        ])

    def _write_controller_metadata(self):
        model_path = cfg.model_path
        digest = None
        if os.path.exists(model_path):
            hasher = hashlib.sha256()
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        payload = {
            'run_id': RUN_ID,
            'method': METHOD_NAME,
            'model_path': model_path,
            'model_sha256': digest,
            'state_dim': cfg.state_dim,
            'action_dim': cfg.action_dim,
            'prediction_mode': cfg.prediction_mode,
        }
        with open(os.path.join(str(RUN_DIR), 'controller_metadata.json'), 'w') as f:
            json.dump(payload, f, indent=2)

    def callback_manual_outcome(self, msg: String):
        allowed = {
            'successful_return','missed_ball','edge_contact','net_failure',
            'out_of_table','robot_reach','detection_failure','unknown'
        }
        parts = msg.data.split('|', 1)
        outcome = parts[0].strip().lower()
        note = parts[1].strip() if len(parts) > 1 else ''
        if outcome not in allowed:
            rospy.logwarn(f"Unknown trial outcome {outcome!r}; allowed={sorted(allowed)}")
            return
        episode = self.pending_outcome_episode
        if episode is None:
            episode = self.last_action_episode if self.last_action_episode is not None else self.epoise
        with open(self.manual_outcome_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                RUN_ID, METHOD_NAME, rospy.Time.now().to_sec(), episode, outcome, note
            ])
        rospy.loginfo(f"[Outcome] episode={episode}, outcome={outcome}, note={note}")
        if self.awaiting_outcome:
            self.awaiting_outcome = False
            self.pending_outcome_episode = None
            self.trigger_next_round()

    def _log_auto_result(self, rewards_info, trajectory_points):
        episode = int(self.epoise)
        if episode in self.logged_result_episodes:
            rospy.logwarn(
                f"[ResultLog] duplicate result suppressed for episode={episode}"
            )
            return False
        reward = rewards_info.get('reward_vec') if rewards_info else None
        values = [np.nan] * 7 if reward is None else list(np.asarray(reward, dtype=float))
        with open(self.auto_result_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                RUN_ID, METHOD_NAME, episode, trajectory_points,
                int(bool(rewards_info and rewards_info.get('success'))),
                int(bool(rewards_info and rewards_info.get('hit_success'))),
                *values
            ])
        self.logged_result_episodes.add(episode)
        return True

    def _log_prediction_failure(self, failure_reason, phase=None):
        """Persist one terminal prediction failure and one invalid result row.

        Transient fit attempts are not written here.  The method is called only
        when the whole round is being terminated, so one episode contributes at
        most one row to prediction_failures.csv and auto_results.csv.
        """
        episode = int(self.epoise)
        if episode in self.logged_failure_episodes:
            return False
        reason = str(failure_reason).strip() or 'unknown_prediction_failure'
        phase_name = str(phase if phase is not None else self.phase)
        with open(self.prediction_failure_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                RUN_ID, METHOD_NAME, OBSERVATION_MODE, episode,
                rospy.Time.now().to_sec(), phase_name, reason,
                len(self.traj_buffer), self.fit_point_count,
                int(self.current_state_vec is not None),
                int(self.current_action_vec is not None), 0
            ])
        self.logged_failure_episodes.add(episode)
        self._log_auto_result(None, len(self.traj_buffer))
        rospy.logwarn(
            f"[PredictionFailure] episode={episode}, phase={phase_name}, "
            f"reason={reason}, trajectory_points={len(self.traj_buffer)}"
        )
        return True

    def _update_success_stats(self, hit_success):
        """Count every completed evaluation round, including prediction failures."""
        if not cfg.enable_success_stats or self.current_epoise < 50:
            return
        self.total_rounds += 1
        if hit_success:
            self.success_hits += 1
        if self.total_rounds % self.stats_interval == 0 or self.current_epoise >= 149:
            rate = self.success_hits / self.total_rounds * 100.0
            rospy.loginfo(
                f"[Stats] Rounds: {self.total_rounds}, "
                f"Success: {self.success_hits}, Rate: {rate:.1f}%"
            )

    def _check_prediction_holdout(self, min_fit_points=8, min_eval_points=5):
        if len(self.traj_buffer) < min_fit_points + min_eval_points:
            return

        pts = np.array([d['p'] for d in self.traj_buffer])
        times = np.array([d['t'] for d in self.traj_buffer])

        fit_pts = pts[:min_fit_points]
        fit_times = times[:min_fit_points]

        eval_pts = pts[min_fit_points:]
        eval_times = times[min_fit_points:]

        fit = PhysicsEngine.fit_parabola(fit_pts, fit_times)
        if fit is None:
            rospy.logwarn("[Holdout] fit failed.")
            return

        pred_pts = []
        for t in eval_times:
            p_pred, _ = PhysicsEngine.get_state(fit, t)
            pred_pts.append(p_pred)

        pred_pts = np.array(pred_pts)
        err = pred_pts - eval_pts

        mean_err = np.mean(err, axis=0)
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        max_abs = np.max(np.abs(err), axis=0)

        rospy.logwarn(
            f"[Holdout] n_fit={len(fit_pts)}, n_eval={len(eval_pts)}, "
            f"mean_err=({mean_err[0]:.3f},{mean_err[1]:.3f},{mean_err[2]:.3f}), "
            f"rmse=({rmse[0]:.3f},{rmse[1]:.3f},{rmse[2]:.3f}), "
            f"max_abs=({max_abs[0]:.3f},{max_abs[1]:.3f},{max_abs[2]:.3f})"
        )

    def check_idle_timeout(self, event):
        with self.lock:
            if self.awaiting_outcome:
                return
            if self.phase == "IDLE" and (rospy.get_time() - self.last_msg_time) > 2.0:
                self.trigger_next_round()

    def callback_ball(self, msg: PointStamped):
        t_abs = msg.header.stamp.to_sec()
        p = np.array([msg.point.x, msg.point.y, msg.point.z])
        with self.lock:
            if self.round_start_abs is None:
                self.round_start_abs = t_abs
            t_rel = t_abs - self.round_start_abs
            self.data_queue.put({'t': t_rel, 'p': p})
            self.last_msg_time = t_abs

    def reset_round(self):
        with self.lock:
            self.traj_buffer = []
            self.current_state_vec = None
            self.current_action_vec = None
            self.true_current_action_vec = None
            self.idx_hit = None
            self.t_plane_crossing = None
            self.fit_net = None
            self.fit_pre_hit = None
            self.fit_point_count = 0
            self.last_prediction_issue = ''
            self.round_start_abs = None
            self.round_start_time = rospy.Time.now().to_sec()
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
        self.phase = "IDLE"

    def process_loop(self):
        rate = rospy.Rate(cfg.control_freq)
        while not rospy.is_shutdown():
            try:
                with self.lock:
                    new_points = []
                    while not self.data_queue.empty():
                        new_points.append(self.data_queue.get())
                    if new_points:
                        self._update_trajectory(new_points)
                        self._run_state_machine()
                    if len(self.traj_buffer) > self.max_traj_length:
                        rospy.logwarn(f"Trajectory too long ({len(self.traj_buffer)} points), forcing end of round.")
                        self.end_round()
                    # ========== 移除计划生成和执行代码 ==========
                    timeout_phases = [
                        "IDLE", "WAIT_PREDICT_PLANE", "WAIT_NET",
                        "PREDICT", "HIT_DETECTED"
                    ]
                    if self.phase in timeout_phases and self.round_start_time is not None:
                        if rospy.Time.now().to_sec() - self.round_start_time > self.round_timeout_duration:
                            timed_out_phase = self.phase
                            rospy.logwarn(
                                f"Round timeout in phase={timed_out_phase}, forcing end_round"
                            )
                            failure_reason = None
                            if timed_out_phase == "WAIT_PREDICT_PLANE":
                                failure_reason = (
                                    self.last_prediction_issue
                                    or 'predict_plane_crossing_timeout'
                                )
                            elif timed_out_phase in ("IDLE", "WAIT_NET"):
                                failure_reason = (
                                    self.last_prediction_issue
                                    or f'{timed_out_phase.lower()}_timeout'
                                )
                            self.end_round(terminal_failure_reason=failure_reason)
            except Exception as e:
                rospy.logerr(f"process_loop exception: {e}")
                self.last_prediction_issue = (
                    f'process_loop_exception:{type(e).__name__}'
                )
                self.end_round(
                    terminal_failure_reason=self.last_prediction_issue
                )
            rate.sleep()

    def _update_trajectory(self, new_points: List[Dict]):
        rows = []
        for pt in new_points:
            v = np.array([0.0, 0.0, 0.0])
            if len(self.traj_buffer) > 0:
                last = self.traj_buffer[-1]
                dt = pt['t'] - last['t']
                if dt > 1e-4:
                    v = (pt['p'] - last['p']) / dt
            self.traj_buffer.append({'t': pt['t'], 'p': pt['p'], 'v': v})
            rows.append([
                RUN_ID, METHOD_NAME, OBSERVATION_MODE, self.epoise, pt['t'],
                pt['p'][0], pt['p'][1], pt['p'][2], v[0], v[1], v[2], self.phase
            ])
        if rows:
            with open(self.trajectory_log_path, 'a', newline='') as f:
                csv.writer(f).writerows(rows)
        if len(self.traj_buffer) > 500:
            self.traj_buffer = self.traj_buffer[-500:]

    def _run_state_machine(self):
        with self.lock:
            if len(self.traj_buffer) < 10: return
            pts = np.array([d['p'] for d in self.traj_buffer])
            vels = np.array([d['v'] for d in self.traj_buffer])
            times = np.array([d['t'] for d in self.traj_buffer])

            if self.phase == "IDLE":
                if np.any(pts[:, 0] < 0) and np.mean(vels[-5:, 0]) > 0.5:
                    self.phase = "WAIT_PREDICT_PLANE"

            if self.phase == "WAIT_PREDICT_PLANE" and self.t_plane_crossing is None:
                if pts[-1, 0] >= self.predict_plane_x and len(pts) >= 4:
                    self._check_prediction_holdout(min_fit_points=8, min_eval_points=5)
                    fit_start = 0 if FIT_WINDOW_SIZE == 0 else max(0, len(pts) - FIT_WINDOW_SIZE)
                    fit_pts = pts[fit_start:]
                    fit_times = times[fit_start:]
                    fit = PhysicsEngine.fit_parabola(fit_pts, fit_times)
                    if fit is not None:
                        self.fit_net = fit
                        self.fit_point_count = len(fit_pts)
                        t_net = PhysicsEngine.solve_time_for_axis(
                            fit, 0, self.predict_plane_x, fit_times[0]
                        )

                        if t_net is None:
                            self.last_prediction_issue = (
                                'cannot_solve_predict_plane_crossing_time'
                            )
                            rospy.logwarn("Cannot solve predict-plane crossing time, keep waiting.")
                            return

                        self.t_plane_crossing = t_net
                        self.last_prediction_issue = ''
                        ok = self._run_rl_inference()
                        if ok:
                            self.phase = "PREDICT"
                        else:
                            reason = (
                                self.last_prediction_issue or 'rl_inference_failed'
                            )
                            rospy.logwarn(
                                f"RL inference failed ({reason}), end this round."
                            )
                            self.end_round(terminal_failure_reason=reason)
                    else:
                        self.last_prediction_issue = 'parabola_fit_failed'
                        rospy.logwarn("Parabola fit failed, waiting for more points")
            elif self.phase == "PREDICT" and self.idx_hit is None:
                if self.t_plane_crossing is None:
                    return
                for i in range(1, len(vels)-1):
                    if times[i] < self.t_plane_crossing:
                        continue
                    if abs(vels[i+1, 0] - vels[i, 0]) > 3.0 and pts[i,0] > 0:
                        self.idx_hit = i+1
                        start_idx = int(np.searchsorted(times, self.t_plane_crossing))
                        if self.idx_hit > start_idx + 4:
                            self.fit_pre_hit = PhysicsEngine.fit_parabola(pts[start_idx:self.idx_hit], times[start_idx:self.idx_hit])
                        self.phase = "HIT_DETECTED"
                        break

    def _apply_safe_hit_range(self, cmd: HitCommand):
        if not ENABLE_SAFE_HIT_RANGE:
            return False

        raw_x = cmd.target_point.x
        raw_y = cmd.target_point.y
        raw_z = cmd.target_point.z

        cmd.target_point.x = float(np.clip(cmd.target_point.x, SAFE_HIT_X_MIN, SAFE_HIT_X_MAX))
        cmd.target_point.y = float(np.clip(cmd.target_point.y, SAFE_HIT_Y_MIN, SAFE_HIT_Y_MAX))
        cmd.target_point.z = float(np.clip(cmd.target_point.z, SAFE_HIT_Z_MIN, SAFE_HIT_Z_MAX))

        if abs(cmd.target_point.x - raw_x) > 1e-6 or \
           abs(cmd.target_point.y - raw_y) > 1e-6 or \
           abs(cmd.target_point.z - raw_z) > 1e-6:
            rospy.logwarn(
                f"[SafeRange] target clipped: "
                f"raw=({raw_x:.3f},{raw_y:.3f},{raw_z:.3f}) -> "
                f"safe=({cmd.target_point.x:.3f},{cmd.target_point.y:.3f},{cmd.target_point.z:.3f})"
            )
            return True
        return False

    def _timing_from_policy(self, timing_value, target_x, state_vec):
        """Return the equivalent k, physical t_delay, and scheduled time.

        In t_delay mode ``timing_value`` is the policy's normalized sixth
        output.  Its physical delay interval is derived from K_RANGE for the
        current ball.  Thus the two interval endpoints correspond exactly:
        the earliest delay maps to k_max and the latest to k_min.
        """
        state_t0 = float(self.fit_net['t0']) if self.fit_net is not None else 0.0
        return resolve_policy_timing(
            mode=TIMING_MODE,
            timing_value=float(timing_value),
            x0=float(state_vec[0]),
            vx0=float(state_vec[3]),
            x_hit=float(target_x),
            state_t0=state_t0,
            predicted_hit_time=float(self.predicted_hit_time),
            k_range=K_RANGE,
        )

    def _log_action(self, state_vec, raw_action, mapped_action, hit_pos, hit_vel,
                    policy_target, executed_target, target_clipped, timing_value,
                    k, t_delay, scheduled_hit_rel, time_to_hit, cmd_vx,
                    command_published, rejection_reason):
        row = [
            RUN_ID, METHOD_NAME, OBSERVATION_MODE, self.epoise,
            rospy.Time.now().to_sec(), *np.asarray(state_vec, dtype=float).tolist(),
            *np.asarray(raw_action, dtype=float).tolist(),
            *np.asarray(mapped_action, dtype=float).tolist(),
            *np.asarray(hit_pos, dtype=float).tolist(),
            *np.asarray(hit_vel, dtype=float).tolist(),
            *np.asarray(policy_target, dtype=float).tolist(),
            *np.asarray(executed_target, dtype=float).tolist(),
            int(bool(target_clipped)), float(timing_value), k, t_delay,
            float(self.predicted_hit_time), scheduled_hit_rel, time_to_hit, cmd_vx,
            int(DEBUG_AIM_AT_PREDICTED_BALL), int(DEBUG_FIXED_POSE),
            int(DEBUG_FIXED_VX), int(ENABLE_SAFE_HIT_RANGE), self.fit_point_count,
            int(bool(command_published)), rejection_reason
        ]
        with open(self.action_log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        self.last_action_episode = self.epoise
        self.episode_pub.publish(Int32(data=int(self.epoise)))

    def _run_rl_inference(self):
        if self.fit_net is None:
            self.last_prediction_issue = 'fit_not_available'
            return False
        t_hit, hit_info = self._extrapolate_hit_time()
        if t_hit is None or hit_info is None:
            self.last_prediction_issue = 'cannot_extrapolate_hit_time'
            rospy.logwarn("Cannot extrapolate hit time.")
            return False
        self.predicted_hit_time = t_hit
        hit_pos, hit_vel = hit_info
        cmd = HitCommand()
        if cfg.prediction_mode == "HIT_PLANE":
            if cfg.is_real_robot == True:
                target_x = np.random.uniform(-0.49, -0.5)
                target_y = np.random.uniform(-0.01, 0.01)
            else:
                target_x = np.random.uniform(-0.3, -0.6)
                target_y = np.random.uniform(-0.2, 0.2)
            target_v = np.random.uniform(6.0, 8.0)
            state_vec = np.concatenate([hit_pos, hit_vel, [target_x, target_y, target_v]])
            self.current_state_vec = state_vec
            action_vec, true_action_vec = self.brain.get_action(state_vec)
            self.current_action_vec = action_vec
            self.true_current_action_vec = true_action_vec
            cmd.target_point.x = hit_pos[0] + self.true_current_action_vec[0]
            cmd.target_point.y = hit_pos[1] + self.true_current_action_vec[1]
            cmd.target_point.z = hit_pos[2] + self.true_current_action_vec[2]

        elif cfg.prediction_mode == "INIT_PLANE":
            coeffs = self.fit_net['coeffs']
            t0 = self.fit_net['t0']
            pos_init, vel_init = PhysicsEngine.get_state(self.fit_net, t0)
            if cfg.is_real_robot == True:
                target_x = np.random.uniform(-0.49, -0.5)
                target_y = np.random.uniform(-0.01, 0.01)
            else:
                target_x = np.random.uniform(-0.3, -0.6)
                target_y = np.random.uniform(-0.2, 0.2)
            target_v = np.random.uniform(6.0, 8.0)
            state_vec = np.concatenate([pos_init, vel_init, [target_x, target_y, target_v]])
            self.current_state_vec = state_vec
            action_vec, true_action_vec = self.brain.get_action(state_vec)
            self.current_action_vec = action_vec
            self.true_current_action_vec = true_action_vec
            cmd.target_point.x = self.true_current_action_vec[0]
            cmd.target_point.y = self.true_current_action_vec[1]
            cmd.target_point.z = self.true_current_action_vec[2]
        else:
            self.last_prediction_issue = 'unsupported_prediction_mode'
            return False
                # ========== Debug: 直接用视觉预测的击球点作为目标 ==========
        policy_target = np.array([
            cmd.target_point.x, cmd.target_point.y, cmd.target_point.z
        ], dtype=float)
        target_clipped = False
        if DEBUG_AIM_AT_PREDICTED_BALL:
            cmd.target_point.x = cfg.hit_plane_x
            cmd.target_point.y = hit_pos[1]
            cmd.target_point.z = hit_pos[2]

            rospy.logwarn(
                f"[DebugAim] Use predicted ball position as target: "
                f"target=({cmd.target_point.x:.3f},{cmd.target_point.y:.3f},{cmd.target_point.z:.3f}), "
                f"original_network_target=({self.true_current_action_vec[0]:.3f},"
                f"{self.true_current_action_vec[1]:.3f},"
                f"{self.true_current_action_vec[2]:.3f})"
            )
        if ENABLE_SAFE_HIT_RANGE:
            target_clipped = self._apply_safe_hit_range(cmd)
        aim_err = np.array([
            cmd.target_point.x - hit_pos[0],
            cmd.target_point.y - hit_pos[1],
            cmd.target_point.z - hit_pos[2]
        ])

        rospy.logwarn(
            f"[AimCheck] ball_hit_pos=({hit_pos[0]:.3f},{hit_pos[1]:.3f},{hit_pos[2]:.3f}), "
            f"cmd_hit_pos=({cmd.target_point.x:.3f},{cmd.target_point.y:.3f},{cmd.target_point.z:.3f}), "
            f"err=({aim_err[0]:.3f},{aim_err[1]:.3f},{aim_err[2]:.3f}), "
            f"err_norm={np.linalg.norm(aim_err):.3f}"
        )
        #计算姿态：
                # 计算姿态
        default_euler = [
            self.brain.default_pose['rx'],
            self.brain.default_pose['ry'],
            self.brain.default_pose['rz']
        ]

        if DEBUG_FIXED_POSE:
            delta_ry = 0.0
            delta_rz = 0.0
            if DEBUG_LINEAR_RZ_BY_Y:
                y_for_rz = float(cmd.target_point.y)

                # y<0 时 delta_rz<0，y>0 时 delta_rz>0
                ratio = np.clip(y_for_rz / DEBUG_RZ_Y_RANGE, -1.0, 1.0)
                delta_rz = ratio * DEBUG_RZ_MAX_DELTA

            final_euler = [
                default_euler[0],
                default_euler[1] + delta_ry,
                default_euler[2] + delta_rz
            ]

        else:
            delta_ry = self.true_current_action_vec[3]
            delta_rz = self.true_current_action_vec[4]
            final_euler = [
                default_euler[0],
                default_euler[1] + delta_ry,
                default_euler[2] + delta_rz
            ]

        quat = self._euler_to_quaternion(
            final_euler[0],
            final_euler[1],
            final_euler[2]
        )
        timing_value = float(self.true_current_action_vec[5])
        k, t_delay, scheduled_hit_rel = self._timing_from_policy(
            timing_value, cmd.target_point.x, state_vec
        )
        if k is None or t_delay is None or scheduled_hit_rel is None:
            self.last_prediction_issue = 'invalid_timing_conversion'
            rospy.logwarn(
                f"Invalid timing conversion for value={timing_value:.6f}"
            )
            return False
        # In t_delay mode the intermediate value above is only the normalized
        # selector.  Store/log the physical delay actually sent to the robot.
        if TIMING_MODE == 't_delay':
            self.true_current_action_vec[5] = float(t_delay)
            timing_value = float(t_delay)
        hit_abs_time = self.round_start_abs + scheduled_hit_rel
        now_abs = rospy.Time.now().to_sec()


        command_published = hit_abs_time > now_abs + 0.08
        rejection_reason = '' if command_published else 'hit_time_too_close_or_expired'
        # ========== 发布击球命令 ==========

        duration = 0.3   # 击球后继续运动时间（秒），可调整
        cmd.target_orientation.x = quat[0]
        cmd.target_orientation.y = quat[1]
        cmd.target_orientation.z = quat[2]
        cmd.target_orientation.w = quat[3]

        cmd.hit_time = hit_abs_time
        cmd.duration = duration

        if DEBUG_FIXED_VX:
            cmd.vec = DEBUG_FIXED_VX_VALUE
        else:
            cmd.vec = float(self.true_current_action_vec[6])
        executed_target = np.array([
            cmd.target_point.x, cmd.target_point.y, cmd.target_point.z
        ], dtype=float)
        self.last_timing_value = timing_value
        self.last_k = k
        self.last_t_delay = t_delay
        self._log_action(
            state_vec, action_vec, self.true_current_action_vec, hit_pos, hit_vel,
            policy_target, executed_target, target_clipped, timing_value, k,
            t_delay, scheduled_hit_rel, hit_abs_time - now_abs, cmd.vec,
            command_published, rejection_reason
        )
        rospy.logwarn(
            f"[Predict] now={now_abs:.3f}, hit_abs={hit_abs_time:.3f}, "
            f"time_to_hit={hit_abs_time-now_abs:.3f}, "
            f"target=({cmd.target_point.x:.3f},{cmd.target_point.y:.3f},{cmd.target_point.z:.3f}), "
            f"pred_ball=({hit_pos[0]:.3f},{hit_pos[1]:.3f},{hit_pos[2]:.3f}), "
            f"vx_cmd={cmd.vec:.3f}, "
            f"timing_value={timing_value:.6f}, k={k}, t_delay={t_delay:.6f}, "
            f"debug_aim={DEBUG_AIM_AT_PREDICTED_BALL}, "
            f"debug_pose={DEBUG_FIXED_POSE}, "
            f"debug_vx={DEBUG_FIXED_VX}"
        )
        if not command_published:
            self.last_prediction_issue = 'hit_time_too_close_or_expired'
            rospy.logwarn(
                f"Predicted hit time too close or expired. hit_abs_time={hit_abs_time:.3f}, now={now_abs:.3f}"
            )
            return False
        #rospy.loginfo(f"Published HitCommand: target=({cmd.target_point.x:.2f},{cmd.target_point.y:.2f},{cmd.target_point.z:.2f}) quat=({quat[0]:.2f},{quat[1]:.2f},{quat[2]:.2f},{quat[3]:.2f}, hit_time={hit_abs_time:.3f} duration={duration} hit_vec={cmd.vec:.3f}")
        self.hit_cmd_pub.publish(cmd)
        self.last_prediction_issue = ''
        return True
    def _extrapolate_hit_time(self):

        if self.fit_net is None:
            return None, None
        if self.t_plane_crossing is None:
            rospy.logwarn("Cannot extrapolate hit time: t_plane_crossing is None.")
            return None, None
        coeffs = self.fit_net['coeffs']
        t0 = self.fit_net['t0']
        a, b, c = coeffs[6], coeffs[7], coeffs[8] - cfg.table_z
        delta = b**2 - 4*a*c
        if delta < 0:
            return None, None
        sqrt_d = np.sqrt(delta)
        trs = [(-b+sqrt_d)/(2*a), (-b-sqrt_d)/(2*a)]
        t_bounce = None
        for tr in trs:
            tg = t0 + tr
            if tg > self.t_plane_crossing:
                vz = 2*a*tr + b
                if vz < 0:
                    t_bounce = tg
                    break
        if t_bounce is None:
            return None, None
        pos_b, vel_b = PhysicsEngine.get_state(self.fit_net, t_bounce)
        pos_p, vel_p = PhysicsEngine.simulate_bounce(pos_b, vel_b)
        if abs(vel_p[0]) < 0.1:
            return None, None
        dt_to_hit = (cfg.hit_plane_x - pos_p[0]) / vel_p[0]
        if dt_to_hit <= 0:
            return None, None
        t_hit = t_bounce + dt_to_hit
        hit_pos = np.array([
            cfg.hit_plane_x,
            pos_p[1] + vel_p[1]*dt_to_hit,
            pos_p[2] + vel_p[2]*dt_to_hit - 0.5*cfg.gravity*dt_to_hit**2
        ])
        hit_vel = np.array([
            vel_p[0],
            vel_p[1],
            vel_p[2] - cfg.gravity*dt_to_hit
        ])
        return t_hit, (hit_pos, hit_vel)

    def _euler_to_quaternion(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [x, y, z, w]

    def end_round(self, terminal_failure_reason=None):
        finished_episode = self.epoise
        should_trigger = False
        with self.lock:
            self.current_epoise += 1
            if terminal_failure_reason is not None:
                self._log_prediction_failure(terminal_failure_reason)
                self._update_success_stats(False)
                self.reset_round()
                should_trigger = True
            elif self.phase == "IDLE":
                self._log_prediction_failure(
                    self.last_prediction_issue or 'idle_timeout_without_valid_trajectory'
                )
                self._update_success_stats(False)
                self.reset_round()
                should_trigger = True
            else:
                if len(self.traj_buffer) < 10:
                    rospy.logwarn("Too few points for reward.")
                    self._log_prediction_failure('too_few_trajectory_points')
                    self._update_success_stats(False)
                    self.reset_round()
                    should_trigger = True
                elif self.current_state_vec is None or self.current_action_vec is None:
                    rospy.logwarn("No state/action to evaluate.")
                    self._log_prediction_failure(
                        self.last_prediction_issue or 'no_state_or_action'
                    )
                    self._update_success_stats(False)
                    self.reset_round()
                    should_trigger = True
                else:
                    times = np.array([d['t'] for d in self.traj_buffer])
                    positions = np.array([d['p'] for d in self.traj_buffer])
                    veclocity = np.array([d['v'] for d in self.traj_buffer])
                    rewardsInfo = self.calculate_reward_detailed(
                        positions, times, veclocity, self.current_state_vec, self.true_current_action_vec, self.predicted_hit_time
                    )
                    self._log_auto_result(rewardsInfo, len(self.traj_buffer))
                    self._update_success_stats(
                        bool(rewardsInfo and rewardsInfo.get('hit_success'))
                    )
                    if rewardsInfo['success']:
                        raw_reward = np.array(rewardsInfo['reward_vec'])
                        weighted_reward = raw_reward * np.array([10, 10, 10, 5, 5, 5, 0.1])
                        buffer_yz = [self.current_state_vec, self.current_action_vec, weighted_reward]
                        rospy.loginfo(f'current_action_vec:{self.current_action_vec}')
                        rospy.loginfo(f'rt:{weighted_reward}')
                        self.save_data(buffer_yz)
                        #target_x = self.current_state_vec[6]
                        #target_y = self.current_state_vec[7]
                        #target_z = cfg.table_z + 0.1
                        #with open("/tmp/pingpong_target.txt", "w") as f:
                            #f.write(f"{target_x},{target_y},{target_z}\n")
                        # ========== 新增：写入奖励分量 ==========
                        r = weighted_reward
                        if self.test_flag:
                            pass
                        else:
                            with open(self.reward_log_path, 'a') as f:
                                f.write(f"{self.epoise},{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n")
                    self.reset_round()
                    should_trigger = True

        if should_trigger:
            self.epoise += 1
            rospy.loginfo(f'epoise:{self.epoise}')
            if self.epoise > 0 and self.epoise % 10 == 0:
                self.brain.load_model()
                rospy.loginfo('reload_model')
            if cfg.is_real_robot and REQUIRE_MANUAL_OUTCOME:
                self.awaiting_outcome = True
                self.pending_outcome_episode = finished_episode
                self.episode_pub.publish(Int32(data=int(finished_episode)))
                rospy.loginfo(
                    f"Waiting for manual outcome for episode {finished_episode} before next serve."
                )
            else:
                self.trigger_next_round()

    def save_data(self, buffer_yz: list):
        if not buffer_yz:
            return
        new_states_arr = np.array(buffer_yz[0], dtype=np.float32)
        new_actions_arr = np.array(buffer_yz[1], dtype=np.float32)
        new_rewards_arr = np.array(buffer_yz[2], dtype=np.float32)
        if cfg.is_real_robot:
            filename = self.data_file
            if os.path.exists(filename):
                data = np.load(filename, allow_pickle=True)
                existing_states = data['states']
                existing_actions = data['actions']
                existing_rewards = data['rewards']
                data.close()
                combined_states = np.concatenate([existing_states, new_states_arr])
                combined_actions = np.concatenate([existing_actions, new_actions_arr])
                combined_rewards = np.concatenate([existing_rewards, new_rewards_arr])
            else:
                combined_states = new_states_arr
                combined_actions = new_actions_arr
                combined_rewards = new_rewards_arr
            np.savez_compressed(filename,
                                states=combined_states,
                                actions=combined_actions,
                                rewards=combined_rewards)
        else:
            filename = os.path.join(DATA_DIR, f"buffer_{self.epoise}.npz")
            np.savez_compressed(filename,
                                states=new_states_arr,
                                actions=new_actions_arr,
                                rewards=new_rewards_arr)
        rospy.loginfo(f"Saved to {filename}")

    def trigger_next_round(self):
        if self.max_episodes is not None and self.epoise >= self.max_episodes:
            rospy.loginfo(f"Reached max episodes {self.max_episodes}, no more rounds.")
            rospy.signal_shutdown("Max episodes reached")
            return
        if cfg.is_real_robot:
            now = rospy.Time.now().to_sec()
            if now - self.last_msg_time < 0.5:   # 0.5秒内收到过球，说明球还在运动
                rospy.loginfo("Ball detected recently, skip auto serve.")
                return
        if cfg.is_real_robot:
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"1")
                    self.serial_port.flush()
                    rospy.loginfo("Trigger sent to Arduino")
                except Exception as e:
                    rospy.logerr(f"Serial write failed: {e}")
        else:
            self.pub_sim_start.publish(Bool(data=True))

    # ==========================================
    # 奖励计算模块
    # ==========================================
    def calculate_reward_detailed(self, positions, times, velocities, state_vec, action_vec, predicted_hit_time):
        hit_y = action_vec[1]
        hit_z = action_vec[2]
        t_delay = self.last_t_delay if self.last_t_delay is not None else action_vec[5]
        target_x = state_vec[6]
        target_y = state_vec[7]
        target_v = state_vec[8]
        reward = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        N = len(positions)
        if N < 10:
            return {'reward_vec': None, 'success': False, 'hit_success':False}

        bounce_idx = _detect_first_bounce(velocities, positions, times)
        if bounce_idx is None:
            return {'reward_vec': None, 'success': False, 'hit_success':False}

        hit_idx = _detect_hit_after_bounce(velocities, bounce_idx)
        if hit_idx is None:
            pts_after = positions[bounce_idx:]
            t_after = times[bounce_idx:]
            fit = PhysicsEngine.fit_parabola(pts_after, t_after)
            if fit is None:
                return {'reward_vec': None, 'success': False, 'hit_success':False}
            t_hit = PhysicsEngine.solve_time_for_axis(fit, axis=0,
                                                      target_val=cfg.hit_plane_x,
                                                      t_min_global=t_after[0])
            if t_hit is None:
                p_hit = positions[-1]
                v_hit = velocities[-1]
                t_hit = times[-1]
            else:
                p_hit, v_hit = PhysicsEngine.get_state(fit, t_hit)
            has_post_data = False
        else:
            pts_after = positions[bounce_idx:hit_idx]
            t_after = times[bounce_idx:hit_idx]
            fit = PhysicsEngine.fit_parabola(pts_after, t_after)
            if fit is None:
                return {'reward_vec': None, 'success': False, 'hit_success':False}
            t_hit = PhysicsEngine.solve_time_for_axis(fit, axis=0,
                                                      target_val=cfg.hit_plane_x,
                                                      t_min_global=t_after[0])
            if t_hit is None:
                p_hit = positions[-1]
                v_hit = velocities[-1]
                t_hit = times[-1]
            else:
                p_hit, v_hit = PhysicsEngine.get_state(fit, t_hit)
            has_post_data = (hit_idx < N - 4)

        reward[0] = abs(p_hit[1] - hit_y)
        reward[1] = abs(p_hit[2] - hit_z)
        reward[2] = abs(t_hit - t_delay - predicted_hit_time)
        rospy.logwarn(
            f"[EvalHit] pred_y={hit_y:.3f}, pred_z={hit_z:.3f}, "
            f"real_y={p_hit[1]:.3f}, real_z={p_hit[2]:.3f}, "
            f"dy={reward[0]:.3f}, dz={reward[1]:.3f}, dt={reward[2]:.3f}"
        )
        hit_success = False
        if has_post_data:
            pts_post = positions[hit_idx:]
            velocities_post = velocities[hit_idx:]
            t_post = times[hit_idx:]
            net_idx = None
            for i in range(len(pts_post)):
                if pts_post[i, 0] <= 0:
                    net_idx = i
                    break
            if net_idx is not None:
                h_net = pts_post[net_idx, 2]
            else:
                h_net = -2.0 + cfg.net_height
            reward[3] = h_net - cfg.net_height

            bounce2_idx = _detect_first_bounce(velocities_post, pts_post, t_post)
            if bounce2_idx is not None:
                p_land = pts_post[bounce2_idx]
                v_land = velocities_post[bounce2_idx]
            else:
                p_land = pts_post[-1]
                v_land = velocities_post[-1]
            if self.test_flag:
                dx = p_land[0] - target_x
                dy = p_land[1] - target_y
                dv = np.linalg.norm(v_land) - target_v
                self._write_csv('target_point.csv',['dx', 'dy', 'dv'],[dx, dy, dv])
            reward[4] = -abs(p_land[0] - target_x)
            reward[5] = -abs(p_land[1] - target_y)
            reward[6] = -abs(np.linalg.norm(v_land) - target_v) * 0.1
            hit_success = (net_idx is not None) and (bounce2_idx is not None)
        #self._write_debug_csv(self.epoise, state_vec[:6], p_hit, t_hit)
        return {'reward_vec': reward, 'success': True, 'hit_success':hit_success}

    def _write_debug_csv(self, episode, init_state, p_hit, t_hit):
        debug_path = os.path.join(self.current_dir, "debug_controller.csv")
        file_exists = os.path.isfile(debug_path)
        with open(debug_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['episode',
                                'fit_x','fit_y','fit_z','fit_vx','fit_vy','fit_vz',
                                'hit_x','hit_y','hit_z','hit_t'])
            writer.writerow([episode] + init_state.tolist() +
                            p_hit.tolist() + [t_hit])
    def _write_csv(self, csv_name, csv_title, csv_Info):
        csv_path = os.path.join(self.current_dir, csv_name)
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(csv_title)
            writer.writerow(csv_Info)
# ==========================================
# 辅助函数（与类无关）
# ==========================================
def _detect_first_bounce(velocities, positions, times):
    N = len(velocities)
    if N < 6:
        return None
    candidates = []
    for i in range(2, N-2):
        if velocities[i-1,2] < -0.1 and velocities[i,2] > 0.1:
            if abs(positions[i,2] - cfg.table_z) < 0.1:
                acc_z = velocities[i,2] - velocities[i-1,2]
                if acc_z > 3.0:
                    candidates.append(i)
    if not candidates:
        for i in range(1, N):
            if velocities[i-1,2] < 0 and velocities[i,2] > 0:
                if abs(positions[i,2] - cfg.table_z) < 0.15:
                    candidates.append(i)
    candidates.sort()
    return candidates[0] if candidates else None

def _detect_hit_after_bounce(velocities, bounce_idx, min_interval=1):
    if bounce_idx is None or bounce_idx >= len(velocities)-1:
        return None
    for i in range(bounce_idx + min_interval, len(velocities)):
        dvx = velocities[i, 0] - velocities[i-1, 0]
        if abs(dvx) > 2.0:
            return i
    return None

# ==========================================
# 5. 入口
# ==========================================
if __name__ == "__main__":
    rospy.init_node('ping_pong_agent')
    agent = PingPongAgent()
    rospy.spin()
