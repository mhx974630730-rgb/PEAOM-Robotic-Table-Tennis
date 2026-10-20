#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import threading
import random
import numpy as np
import rospy
import pybullet as p
import pybullet_data
from geometry_msgs.msg import PointStamped
from mhx_msg_pkg.msg import HitCommand
from std_msgs.msg import Bool
import time
import csv, os
from experiment_config import (
    OBSERVATION_MODE, RESIDUAL_NOISE_STD, RAW_NOISE_STD,
    LINEAR_CORRECTION_PATH, RANDOM_SEED, RUN_ID, RUN_DIR, METHOD_NAME,
    ENABLE_Q2_TRAJECTORY_LOG, Q2_PREDICT_PLANE_X, Q2_REQUIRED_SAMPLES,
    PROJECT_DIR, ensure_run_dir, write_metadata,
)


def apply_observation_model(actual_pos, mode, A, b, rng):
    """Generate exactly one mutually exclusive synthetic observation."""
    actual_pos = np.asarray(actual_pos, dtype=np.float64)
    if mode == 'ideal':
        return actual_pos.copy()
    if mode == 'gaussian_residual':
        return actual_pos + rng.normal(0.0, RESIDUAL_NOISE_STD)
    if mode == 'gaussian_raw':
        return actual_pos + rng.normal(0.0, RAW_NOISE_STD)
    mapped = actual_pos @ A.T + b
    if mode == 'peaom':
        return mapped
    if mode == 'peaom_residual':
        return mapped + rng.normal(0.0, RESIDUAL_NOISE_STD)
    raise RuntimeError(f"Unsupported observation mode: {mode}")
# ==================== 主仿真类 ====================
class BallSimulation:
    def __init__(self):
        # Incoming-ball generation uses Python's random module, whereas the
        # observation model uses NumPy's Generator.  Seed both explicitly.
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        rospy.init_node('ball_simulation', anonymous=True)
        self.pub = rospy.Publisher('/ball_position', PointStamped, queue_size=1)
        self.sub_hit_cmd = rospy.Subscriber('/controller_node/action_publisher', HitCommand, self.hit_command_callback)
        self.sub_reset = rospy.Subscriber('/sim/game_start', Bool, self.reset_callback)
        self.observation_mode = OBSERVATION_MODE
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.A = np.eye(3, dtype=np.float64)
        self.b = np.zeros(3, dtype=np.float64)
        if self.observation_mode in {'peaom', 'peaom_residual'}:
            if not LINEAR_CORRECTION_PATH.exists():
                raise FileNotFoundError(
                    f"PEAOM requires linear mapping: {LINEAR_CORRECTION_PATH}"
                )
            linear_data = np.load(str(LINEAR_CORRECTION_PATH))
            self.A = np.asarray(linear_data['A'], dtype=np.float64)
            self.b = np.asarray(linear_data['b'], dtype=np.float64)
        ensure_run_dir()
        write_metadata()
        self.observation_log_path = os.path.join(str(RUN_DIR), 'sim_observations.csv')
        if not os.path.exists(self.observation_log_path) or os.path.getsize(self.observation_log_path) == 0:
            with open(self.observation_log_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'run_id','method','observation_mode','episode','t_ros',
                    'true_x','true_y','true_z','obs_x','obs_y','obs_z',
                    'err_x','err_y','err_z'
                ])

        # Reviewer 2 Q2: dedicated incoming-trajectory log. Rows are buffered
        # in memory and written only after the TRUE ball position first
        # crosses the prediction plane in the +x direction. Consequently the
        # file contains neither returned trajectories nor incomplete rounds.
        self.enable_q2_trajectory_log = ENABLE_Q2_TRAJECTORY_LOG
        self.q2_predict_plane_x = Q2_PREDICT_PLANE_X
        self.q2_required_samples = Q2_REQUIRED_SAMPLES
        self.q2_log_path = os.path.join(
            str(RUN_DIR), 'q2_true_trajectory_points.csv'
        )
        if self.enable_q2_trajectory_log:
            if not os.path.exists(self.q2_log_path) or os.path.getsize(self.q2_log_path) == 0:
                with open(self.q2_log_path, 'w', newline='') as f:
                    csv.writer(f).writerow([
                        'run_id','method','observation_mode','episode',
                        'sample_idx','t_ros','t_rel','predict_plane_x',
                        'true_x','true_y','true_z',
                        'true_vx','true_vy','true_vz',
                        'obs_x','obs_y','obs_z',
                        'err_x','err_y','err_z','is_crossing_sample'
                    ])

        # These variables must exist before the initial reset() call below.
        self.sim_episode = 0
        self.q2_recording = False
        self.q2_sample_idx = 0
        self.q2_start_time = None
        self.q2_last_true_x = None
        self.q2_rows = []

        # PyBullet 设置
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        urdf_folder = PROJECT_DIR / "urdf_folder"
        self.table_id = p.loadURDF(str(urdf_folder / "tinnestable.urdf"), basePosition=[0, 0, -0.68])
        base_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.arm_id = p.loadURDF(str(urdf_folder / "testarm.urdf"), [1.8, 0, -0.68], base_orn)
        self.ball_id = p.loadURDF(str(urdf_folder / "ball.urdf"), basePosition=[0, 0, 9.3])

        p.changeDynamics(self.ball_id, -1, restitution=0.85)
        p.changeDynamics(self.arm_id, -1, restitution=0.95)
        p.changeDynamics(self.table_id, -1, restitution=0.95)

        # 可视化标记：红色球 = 击球点（来自HitCommand），绿色球 = 奖励目标落点（从文件读取）
        self.red_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
        self.red_marker_id = p.createMultiBody(baseVisualShapeIndex=self.red_marker, basePosition=[0, 0, 0])
        self.green_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1])
        self.green_marker_id = p.createMultiBody(baseVisualShapeIndex=self.green_marker, basePosition=[0, 0, 0])
        self.last_target_read = 0

        # 机械臂控制相关
        self.num_joints = p.getNumJoints(self.arm_id)
        self.controllable_joints = []
        self.joint_names = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.arm_id, i)
            if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joints.append(i)
                self.joint_names.append(info[1].decode('utf-8'))
        rospy.loginfo(f"可控关节索引: {self.controllable_joints}")

        self.ee_link_index = self._get_ee_link_index()
        self.pailink = self.ee_link_index
        rospy.loginfo(f"末端执行器 link 索引: {self.ee_link_index}")

        # 关节速度限制（rad/s）: 前三个 120°/s，后三个 180°/s（放大20倍）
        DEG_TO_RAD = np.pi / 180.0
        MAX_VEL_1 = 120 * DEG_TO_RAD * 20
        MAX_VEL_2 = 180 * DEG_TO_RAD * 20
        self.max_velocities = []
        for idx, _ in enumerate(self.controllable_joints):
            if idx < 3:
                self.max_velocities.append(MAX_VEL_1)
            else:
                self.max_velocities.append(MAX_VEL_2)

        self.lock = threading.RLock()
        self.target_joint_positions = None   # 用于非击球阶段的位置保持

        # 仿真参数
        self.dt = 1.0 / 240.0
        self.control_rate = 240
        self.control_period = 1.0 / self.control_rate
        self.last_control_time = rospy.Time.now().to_sec()
        self.last_pub_time = rospy.Time.now()
        self.pub_interval = rospy.Duration(1.0 / 60.0)

        # 初始化机械臂姿态
        self.reset()
        # 关节偏移量和初始角度（与之前保持一致）
        joint_offsets_rad = [0.0, 1.57, 1.57, 0.0, 1.57, -1.75, 0.0]
        self.joint_offsets_rad = joint_offsets_rad
        actual_initial_deg = [0.0, 180.-25.766, -95.191, 102.389, -187.474, -60.951, -43.988]
        #actual_initial_rad = [0.0, 2.27, -1.59, 1.73, -0.14, -1.14, -0.96]

        urdf_initial_rad = np.radians(actual_initial_deg) + self.joint_offsets_rad
        #urdf_initial_rad = np.array(actual_initial_rad) + np.array(self.joint_offsets_rad)
        #urdf_initial_rad = actual_initial_rad
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(urdf_initial_rad):
                p.resetJointState(self.arm_id, joint_idx, urdf_initial_rad[i])

        time.sleep(0.5)
        rospy.loginfo("Ball simulation started")

        # 击球阶段速度控制相关变量
        self.approach_active = False          # 预定位阶段标志
        self.approach_target_angles = None    # 准备位置的关节角度
        self.swing_active = False             # 击球阶段标志
        self.swing_vx = 0.0                  # 击球速度（x方向）
        self.swing_duration = 0.0            # 击球持续时间
        self.swing_start_time = 0.0          # 击球开始绝对时间
        self.swing_end_time = 0.0            # 击球结束绝对时间

        # 工作空间边界（仅记录，不干预）
        self.workspace_limits = {
            'x_min': 0.8, 'x_max': 1.3,
            'y_min': -0.4, 'y_max': 0.4,
            'z_min': 0.0, 'z_max': 0.8
        }
        self.out_of_workspace_logged = False
        self.approach_target_pos = None    # 预定位目标点
        self.approach_start_time = 0.0
        self.swing_start_time = 0.0
        #debug
        self.swing_real_start = None
        self.record_init_state = None
        self.cross_recorded = False
        self.last_ball_x = 0.0
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def _get_ee_link_index(self):
        link_name_to_index = {}
        for i in range(self.num_joints):
            info = p.getJointInfo(self.arm_id, i)
            link_name = info[12].decode('utf-8')
            link_name_to_index[link_name] = i
        for name in link_name_to_index:
            if 'racket' in name.lower():
                return link_name_to_index[name]
        rospy.logwarn("未找到明确的末端 link，使用最后一个关节的 child link")
        return self.num_joints - 1

    def reset(self):
        """随机复位球的位置和速度"""
        if self.enable_q2_trajectory_log and self.q2_rows:
            rospy.logwarn(
                f"[Q2Log] Discard incomplete episode {self.sim_episode}: "
                f"{len(self.q2_rows)} samples, prediction plane not crossed"
            )

        pos_x = -0.8 + (random.random() - 0.5) * 0.05
        pos_y = (random.random() - 0.5) * 0.3
        pos_z = 0.5 + (random.random() - 0.5) * 0.10
        pos = [pos_x, pos_y, pos_z]
        p.resetBasePositionAndOrientation(self.ball_id, pos, [0, 0, 0, 1])
        vx = 3.9 + (random.random() - 0.5) * 0.2
        vy = -0.1 * (pos_y + (random.random() - 0.5) * 2)
        vz = random.random() * 0.1
        p.resetBaseVelocity(self.ball_id, linearVelocity=[vx, vy, vz])
        # 重置击球标志
        self.target_joint_positions = None
        self.swing_active = False
        self.swing_joint_angles = []
        self.swing_joint_velocities = []
        self.swing_queue_index = 0

        self.record_init_state = np.array([pos_x, pos_y, pos_z, vx, vy, vz])
        self.cross_recorded = False
        self.init_x = pos_x   # 初始时球的 x 坐标

        # Start a fresh Q2 incoming segment. The reset position is used as the
        # previous true x value so that the first crossing cannot be missed.
        self.q2_recording = self.enable_q2_trajectory_log
        self.q2_sample_idx = 0
        self.q2_start_time = None
        self.q2_last_true_x = pos_x
        self.q2_rows = []
        rospy.loginfo(f"球复位:{pos_x, pos_y, pos_z, vx, vy, vz}")

    def _record_q2_sample(self, now, actual_pos, true_velocity, obs_pos, error):
        """Buffer one sample and commit the segment at the prediction plane."""
        if not self.q2_recording:
            return

        t_ros = now.to_sec()
        if self.q2_start_time is None:
            self.q2_start_time = t_ros
        t_rel = t_ros - self.q2_start_time

        true_x = float(actual_pos[0])
        crossed = (
            self.q2_last_true_x is not None
            and self.q2_last_true_x < self.q2_predict_plane_x
            and true_x >= self.q2_predict_plane_x
        )

        self.q2_rows.append([
            RUN_ID, METHOD_NAME, self.observation_mode, self.sim_episode,
            self.q2_sample_idx, t_ros, t_rel, self.q2_predict_plane_x,
            *actual_pos.tolist(), *true_velocity.tolist(),
            *obs_pos.tolist(), *error.tolist(), int(crossed)
        ])
        self.q2_sample_idx += 1
        self.q2_last_true_x = true_x

        if crossed:
            sample_count = len(self.q2_rows)
            with open(self.q2_log_path, 'a', newline='') as f:
                csv.writer(f).writerows(self.q2_rows)

            if sample_count < self.q2_required_samples:
                rospy.logwarn(
                    f"[Q2Log] episode={self.sim_episode} committed with "
                    f"{sample_count} samples (< required "
                    f"{self.q2_required_samples})"
                )
            else:
                rospy.loginfo(
                    f"[Q2Log] episode={self.sim_episode} committed with "
                    f"{sample_count} samples"
                )

            self.q2_rows = []
            self.q2_recording = False

    def reset_callback(self, msg):
        if msg.data:
            self.sim_episode += 1
            self.reset()

    def hit_command_callback(self, msg):
        """收到击球命令（速度控制模式）"""
        target_point = np.array([msg.target_point.x, msg.target_point.y, msg.target_point.z])
        target_orn = [msg.target_orientation.x, msg.target_orientation.y, msg.target_orientation.z, msg.target_orientation.w]
        hit_abs_time = msg.hit_time
        duration = msg.duration
        vec = msg.vec
        rospy.loginfo(f"HitCommand: target=({target_point[0]:.2f},{target_point[1]:.2f},{target_point[2]:.2f}) quat=({target_orn[0]:.2f},{target_orn[1]:.2f},{target_orn[2]:.2f},{target_orn[3]:.2f}, hit_time={hit_abs_time:.3f} duration={duration} hit_vec={vec:.3f}")
        # 更新红色小球位置（击球点）
        p.resetBasePositionAndOrientation(self.red_marker_id, target_point, [0,0,0,1])

        now_abs = rospy.Time.now().to_sec()
        dt = hit_abs_time - now_abs
        if dt <= 0:
            rospy.logwarn(f"Hit time already passed (dt={dt:.3f}), ignoring command")
            return

        # 获取当前末端位置和姿态（姿态保持不变）
        link_state = p.getLinkState(self.arm_id, self.ee_link_index, computeLinkVelocity=0)
        current_pos = np.array(link_state[4])
        current_orn = link_state[5]

        # 预定位目标点：击球点后方 0.1 米（可根据需要调整）
        approach_offset = 0.18
        approach_pos = target_point.copy()
        approach_pos[0] += approach_offset

        approach_pos[0] = np.clip(approach_pos[0], self.workspace_limits['x_min'], self.workspace_limits['x_max'])

        # 使用 IK 求解准备位置的关节角度
        approach_angles = p.calculateInverseKinematics(
            self.arm_id,
            self.ee_link_index,
            approach_pos.tolist(),
            target_orn
        )
        if approach_angles is None:
            rospy.logwarn("Cannot solve IK for approach position, aborting")
            return
                # 记录预定位信息
        self.approach_target_angles = [approach_angles[idx] for idx in self.controllable_joints]
        self.approach_active = True
        self.approach_start_time = now_abs

        # 记录击球阶段参数
        self.swing_vx = vec
        self.swing_duration = duration
        self.swing_start_time = hit_abs_time   # 击球阶段开始时间
        self.swing_end_time = hit_abs_time + duration

        rospy.loginfo(f"Approach to {approach_pos}, then swing at {hit_abs_time:.3f} with vx={vec:.2f}")

    def run(self):
        rate = rospy.Rate(1.0 / self.dt)
        table_bounds = {'x_min': -1.0, 'x_max': 1.0, 'y_min': -0.4, 'y_max': 0.4, 'z_min': -0.05, 'z_max': 2.0}
        see_flag = 0

        while not rospy.is_shutdown():
            p.stepSimulation()
            #debug
            pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            if (self.record_init_state is not None and
                not self.cross_recorded and
                self.last_ball_x < 1.1 and pos[0] >= 1.1):
                #self._write_env_csv(pos)
                self.cross_recorded = True
            self.last_ball_x = pos[0]
            now_sec = rospy.Time.now().to_sec()

            # 定期更新绿色小球（奖励目标落点）
            now_time = time.time()
            if now_time - self.last_target_read > 0.2:
                self.last_target_read = now_time
                try:
                    with open("/tmp/pingpong_target.txt", "r") as f:
                        x, y, z = map(float, f.read().strip().split(','))
                        p.resetBasePositionAndOrientation(self.green_marker_id, [x, y, z], [0,0,0,1])
                except (FileNotFoundError, ValueError):
                    pass

            # ========== 机械臂控制 ==========
            if now_sec - self.last_control_time >= self.control_period:
                self.last_control_time = now_sec

                # 1. 预定位阶段
                if self.approach_active and now_sec <= self.swing_start_time - 0.02:
                    # 使用位置控制移动到准备位置
                    for i, idx in enumerate(self.controllable_joints):
                        p.setJointMotorControl2(
                            self.arm_id, idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=self.approach_target_angles[i],
                            targetVelocity=0.0,
                            force=5000.0,
                            maxVelocity=self.max_velocities[i]
                        )
                    # 检查是否到达准备位置（误差阈值 0.01 rad）
                    current_angles = [p.getJointState(self.arm_id, idx)[0] for idx in self.controllable_joints]
                    #rospy.loginfo(f"current_angles:{current_angles}")
                    #rospy.loginfo(f"target_angles{self.approach_target_angles}")
                    #self.approach_active = False
                    #if all(abs(current_angles[i] - self.approach_target_angles[i]) < 0.01 for i in range(len(self.controllable_joints))):
                    #    rospy.loginfo("Approach completed, ready for swing")
                    #    self.approach_active = False
                    #    # 可选：等待击球时刻，但不需要额外控制，让机械臂保持位置
                    #continue   # 跳过后续控制，等预定位完成
                                # 2. 击球阶段（速度控制）
                elif self.swing_vx != 0 and now_sec >= self.swing_start_time - 0.02:
                    # 第一次进入 swing 时记录
                    if self.swing_real_start is None:
                        # 可选：检查预定位是否到位，若误差太大，延迟开始并告警
                        current_angles = [p.getJointState(self.arm_id, idx)[0] for idx in self.controllable_joints]
                        angle_errors = [abs(current_angles[i] - self.approach_target_angles[i]) for i in range(len(self.controllable_joints))]
                        if max(angle_errors) > 0.05:   # 阈值 0.05 rad
                            rospy.logwarn_throttle(0.2, f"Approach not complete, max angle error: {max(angle_errors):.3f}")
                            # 可选择跳过这次 swing 或继续尝试，这里选择继续但打印警告
                        self.swing_real_start = now_sec

                    # 获取当前末端速度
                    link_state = p.getLinkState(self.arm_id, self.ee_link_index, computeLinkVelocity=1)
                    current_lin = np.array(link_state[6])
                    current_ang = np.array(link_state[7])
                    v_actual = np.hstack([current_lin, current_ang])
                    #rospy.loginfo(f'current_lin{current_lin}')
                    # 期望速度
                    v_des = np.array([-self.swing_vx, 0.0, 0.0, 0.0, 0.0, 0.0])
                    v_error = v_des - v_actual

                    # 当前关节角度
                    current_angles = [p.getJointState(self.arm_id, idx)[0] for idx in self.controllable_joints]
                    zero_vels = [0]*len(current_angles)
                    lin_jac, ang_jac = p.calculateJacobian(
                        self.arm_id, self.ee_link_index, [0,0,0],
                        current_angles, zero_vels, zero_vels
                    )
                    J = np.vstack([lin_jac, ang_jac])  # (6,7)

                    # 1) 前馈部分：开环直接计算达到期望速度需要的关节速度（阻尼伪逆）
                    lambda_ff = 0.05   # 前馈阻尼
                    gain_ff = 1.0
                    try:
                        JJT_ff = J @ J.T + lambda_ff**2 * np.eye(6)
                        delta_ff = np.linalg.solve(JJT_ff, v_des)   # 目标速度映射到关节空间
                        joint_vel_ff = (J.T @ delta_ff) * gain_ff
                    except np.linalg.LinAlgError:
                        joint_vel_ff = np.linalg.pinv(J) @ v_des

                    # 2) 反馈部分：对误差进行小幅修正
                    lambda_fb = 0.1    # 反馈阻尼（稍大以防止振荡）
                    gain_fb = 0.6      # 反馈增益
                    try:
                        JJT_fb = J @ J.T + lambda_fb**2 * np.eye(6)
                        delta_fb = np.linalg.solve(JJT_fb, v_error)
                        joint_vel_fb = (J.T @ delta_fb) * gain_fb
                    except np.linalg.LinAlgError:
                        joint_vel_fb = np.linalg.pinv(J) @ v_error * gain_fb

                    # 合成
                    joint_vel = joint_vel_ff + joint_vel_fb

                    # 关节速度变化率限制（防止突变）
                    if not hasattr(self, '_last_joint_vel'):
                        self._last_joint_vel = np.zeros(len(self.controllable_joints))
                    max_delta = 2.0   # 允许较大变化，因为期望速度可能跳跃
                    delta_vel = joint_vel - self._last_joint_vel
                    delta_vel = np.clip(delta_vel, -max_delta, max_delta)
                    joint_vel = self._last_joint_vel + delta_vel
                    self._last_joint_vel = joint_vel.copy()

                    # 硬限幅
                    for i in range(len(joint_vel)):
                        joint_vel[i] = np.clip(joint_vel[i], -self.max_velocities[i], self.max_velocities[i])

                    # 下发指令
                    for i, idx in enumerate(self.controllable_joints):
                        p.setJointMotorControl2(
                            self.arm_id, idx,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=float(joint_vel[i]),
                            force=5000.0
                        )
                    # 检查击球阶段是否结束
                    if now_sec >= self.swing_start_time + self.swing_duration:
                        if self.swing_real_start is not None:
                            real_duration = now_sec - self.swing_real_start
                            #rospy.loginfo(f"Swing finished, real duration: {real_duration:.3f}s (expected {self.swing_duration:.3f}s)")
                            self.swing_real_start = None
                        #rospy.loginfo("Swing finished")
                        self.swing_vx = 0
                        # 切换回位置保持
                        current_angles = [p.getJointState(self.arm_id, idx)[0] for idx in self.controllable_joints]
                        with self.lock:
                            self.target_joint_positions = current_angles
                else:
                    with self.lock:
                        if self.target_joint_positions is not None:
                            #rospy.loginfo("Pos_controll")
                            for i, idx in enumerate(self.controllable_joints):
                                p.setJointMotorControl2(
                                    self.arm_id, idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.target_joint_positions[i],
                                    targetVelocity=0.0,
                                    force=5000.0,
                                    maxVelocity=self.max_velocities[i]
                                )

            # ========== 球出界检测与位置发布 ==========
            pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            if (pos[2] < table_bounds['z_min'] or pos[2] > table_bounds['z_max'] or
                pos[0] < table_bounds['x_min'] or pos[0] > table_bounds['x_max'] or
                pos[1] < table_bounds['y_min'] or pos[1] > table_bounds['y_max']):
                if see_flag == 0:
                    see_flag = 1
                # 可在此记录球出界事件
            else:
                see_flag = 0
                now = rospy.Time.now()
                if now - self.last_pub_time >= self.pub_interval:
                    actual_pos = np.array(pos, dtype=np.float32)
                    true_velocity, _ = p.getBaseVelocity(self.ball_id)
                    true_velocity = np.array(true_velocity, dtype=np.float32)
                    obs_pos = apply_observation_model(
                        actual_pos, self.observation_mode, self.A, self.b, self.rng
                    )
                    error = obs_pos - actual_pos
                    with open(self.observation_log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            RUN_ID, METHOD_NAME, self.observation_mode, self.sim_episode,
                            now.to_sec(), *actual_pos.tolist(), *obs_pos.tolist(), *error.tolist()
                        ])
                    self._record_q2_sample(
                        now, actual_pos, true_velocity, obs_pos, error
                    )
                    msg = PointStamped()
                    msg.header.stamp = now
                    msg.header.frame_id = "world"
                    msg.point.x = obs_pos[0]
                    msg.point.y = obs_pos[1]
                    msg.point.z = obs_pos[2]
                    self.pub.publish(msg)
                    #rospy.loginfo(f"实际球位置序列: {pos}")
                    #rospy.loginfo(f"观测球位置序列: {obs_pos}")
                    self.last_pub_time = now

            rate.sleep()
    def _write_env_csv(self, cross_pos):
        csv_path = os.path.join(self.current_dir, "debug_env.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['episode',
                                'init_x','init_y','init_z','init_vx','init_vy','init_vz',
                                'cross_x','cross_y','cross_z'])
            writer.writerow([self.sim_episode] + self.record_init_state.tolist() + list(cross_pos))

if __name__ == '__main__':
    try:
        sim = BallSimulation()
        sim.run()
    except rospy.ROSInterruptException:
        pass
