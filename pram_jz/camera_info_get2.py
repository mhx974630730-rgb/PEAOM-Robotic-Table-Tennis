#!/usr/bin/env python3
# coding=utf-8

import os
os.environ["OPENCV_GUI_BACKEND"] = "GTK"

import cv2
import numpy as np
import mvsdk
import platform
import time
import threading
import queue
import sys
import json
from collections import deque
from aim_2d_test import aim_dec
from dataclasses import dataclass
from typing import Tuple, List
import tkinter as tk
from tkinter import simpledialog, messagebox
import datetime

# ------------------------------- 数据结构 ---------------------------------
@dataclass
class FrameData:
    left_frame: np.ndarray
    right_frame: np.ndarray
    timestamp: float
    frame_id: int
    pl: Tuple[int, int] = (-1, -1)
    pr: Tuple[int, int] = (-1, -1)

# ------------------------------- 帧同步 ---------------------------------
class FrameSyncBuffer:
    def __init__(self, max_size=10, time_threshold=0.01):
        self.left_buffer = deque(maxlen=max_size)
        self.right_buffer = deque(maxlen=max_size)
        self.time_threshold = time_threshold

    def add_left(self, frame, timestamp, frame_id):
        self.left_buffer.append((frame, timestamp, frame_id))

    def add_right(self, frame, timestamp, frame_id):
        self.right_buffer.append((frame, timestamp, frame_id))

    def get_synchronized_pair(self):
        if not self.left_buffer or not self.right_buffer:
            return None, None, None, None, None, None
        best_match = None
        best_diff = float('inf')
        for lf, lt, lid in self.left_buffer:
            for rf, rt, rid in self.right_buffer:
                td = abs(lt - rt)
                if td < best_diff and td < self.time_threshold:
                    best_diff = td
                    best_match = (lf, rf, lt, rt, lid, rid)
        if best_match:
            self.left_buffer.clear()
            self.right_buffer.clear()
            return best_match
        return None, None, None, None, None, None

# ------------------------------- 相机采集线程 ---------------------------------
class CameraProcessor:
    def __init__(self, cam_def, display_queue=None, processing_queue=None):
        self.cam_def = cam_def
        self.display_queue = display_queue
        self.processing_queue = processing_queue
        self.running = True
        self.sync_buffer = FrameSyncBuffer(max_size=5)

    def run(self):
        left_id = 0
        right_id = 0
        while self.running:
            try:
                frames = self.cam_def.get_frame()
                if frames[0] is None or frames[1] is None:
                    time.sleep(0.001)
                    continue
                now = time.time()
                self.sync_buffer.add_left(frames[0], now, left_id)
                left_id += 1
                self.sync_buffer.add_right(frames[1], now, right_id)
                right_id += 1
                lf, rf, lt, rt, lid, rid = self.sync_buffer.get_synchronized_pair()
                if lf is not None and rf is not None:
                    fd = FrameData(lf, rf, (lt+rt)/2, lid)
                    if self.processing_queue:
                        try:
                            self.processing_queue.put(fd, block=False)
                        except queue.Full:
                            try:
                                self.processing_queue.get_nowait()
                                self.processing_queue.put(fd, block=False)
                            except queue.Empty:
                                pass
                    if self.display_queue:
                        try:
                            self.display_queue.put(fd, block=False)
                        except queue.Full:
                            try:
                                self.display_queue.get_nowait()
                                self.display_queue.put(fd, block=False)
                            except queue.Empty:
                                pass
            except Exception as e:
                print(f"相机采集错误: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False

# ------------------------------- 检测线程 ---------------------------------
class DetectionProcessor:
    def __init__(self, cam_params, aimdec, processing_queue, display_queue=None):
        self.cam_params = cam_params
        self.aimdec = aimdec
        self.processing_queue = processing_queue
        self.display_queue = display_queue
        self.running = True
        self.latest = None
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            try:
                fd = self.processing_queue.get(timeout=0.001)
            except queue.Empty:
                time.sleep(0.001)
                continue
            pl = self.aimdec.yellow_dec(fd.left_frame)
            pr = self.aimdec.yellow_dec(fd.right_frame)
            fd.pl = pl
            fd.pr = pr
            if pl[0] != -1 and pr[0] != -1:
                world_pt = self.cam_params.triangulate_to_world(pl, pr)
                world_pt[2] = -world_pt[2]
                with self.lock:
                    self.latest = {
                        'pl': pl,
                        'pr': pr,
                        'triangulated': world_pt.copy(),
                        'timestamp': fd.timestamp
                    }
            else:
                with self.lock:
                    self.latest = None
            if self.display_queue:
                try:
                    self.display_queue.put(fd, block=False)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put(fd, block=False)
                    except queue.Empty:
                        pass

    def get_latest_detection(self):
        with self.lock:
            if self.latest is None:
                return None
            return {
                'pl': self.latest['pl'],
                'pr': self.latest['pr'],
                'triangulated': self.latest['triangulated'].copy(),
                'timestamp': self.latest['timestamp']
            }

    def stop(self):
        self.running = False

# ------------------------------- 双目相机参数 ---------------------------------
class StereoCameraParams:
    def __init__(self, rel_npz_file='stereo_calibration_params.npz',
                 abs_npz_file='absolute_extrinsic_params_standard.npz'):
        rel = np.load(rel_npz_file)
        self.mtx_left = rel['mtx_left']
        self.dist_left = rel['dist_left']
        self.mtx_right = rel['mtx_right']
        self.dist_right = rel['dist_right']
        self.R_rel = rel['R']
        self.T_rel = rel['T']
        self.image_size = tuple(rel['image_size'])

        abs_data = np.load(abs_npz_file)
        if 'units' in abs_data and abs_data['units'] == 'meters':
            print("✓ 加载标准版绝对外参（单位：米）")
        else:
            print("⚠️ 外参文件可能不是标准版，确保单位一致")
        self.R_abs = abs_data['R_left']
        self.T_abs = abs_data['T_left'].flatten().reshape(3, 1)
        self.cam_pos_left = abs_data['cam_pos_left'].flatten()
        self.R_inv = np.linalg.inv(self.R_abs)

        self._compute_projection_matrices()
        self._compute_rectification_maps()
        print(f"成功加载相机参数: 图像尺寸 {self.image_size}, "
              f"基线距离 {np.linalg.norm(self.T_rel)*1000:.1f} mm")

    def _compute_projection_matrices(self):
        self.P_left = np.zeros((3, 4))
        self.P_left[:, :3] = self.mtx_left
        self.P_right = np.hstack((self.R_rel, self.T_rel))
        self.P_right = self.mtx_right @ self.P_right

    def _compute_rectification_maps(self):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.mtx_left, self.dist_left, self.mtx_right, self.dist_right,
            self.image_size, self.R_rel, self.T_rel, alpha=0)
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, R1, P1, self.image_size, cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, R2, P2, self.image_size, cv2.CV_32FC1)

    def triangulate_point(self, left_point, right_point):
        left_pt = np.array([[left_point[0], left_point[1]]], dtype=np.float32)
        right_pt = np.array([[right_point[0], right_point[1]]], dtype=np.float32)
        pts4d = cv2.triangulatePoints(self.P_left, self.P_right, left_pt.T, right_pt.T)
        return (pts4d[:3] / pts4d[3]).flatten()

    def camera_to_world(self, pt_cam):
        return self.R_inv @ (pt_cam - self.T_abs.flatten())

    def triangulate_to_world(self, left_point, right_point):
        return self.camera_to_world(self.triangulate_point(left_point, right_point))

# ------------------------------- 相机驱动封装 ---------------------------------
class camera_def(object):
    def __init__(self):
        self.DevList = mvsdk.CameraEnumerateDevice()
        self.nDev = len(self.DevList)
        if self.nDev < 2:
            raise RuntimeError(f"Two cameras are required, but {self.nDev} were found")
        self.frame_count = 0
        self.last_fps_time = time.time()

    def open_camera(self, pg_time):
        DevInfo_list = [None, None]
        for i, DevInfo in enumerate(self.DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        for i in range(self.nDev):
            port_type = self.DevList[i].acPortType
            if b'198.168.58.82' in port_type:
                DevInfo_list[0] = self.DevList[i]
            elif b'197.168.58.224' in port_type:
                DevInfo_list[1] = self.DevList[i]
            else:
                print(f"未知相机端口类型: {port_type}")
        if any(device is None for device in DevInfo_list):
            raise RuntimeError(
                "Could not identify both configured cameras. Update the two "
                "port identifiers in pram_jz/camera_info_get2.py."
            )
        self.nDev = 2
        print(f"找到 {self.nDev} 个相机")
        self.hCamera_list = [0, 0]
        self.pFrameBuffer_list = [0, 0]
        for i, device_info in enumerate(DevInfo_list):
            try:
                hCamera = mvsdk.CameraInit(device_info, -1, -1)
            except mvsdk.CameraException as e:
                raise RuntimeError(
                    "CameraInit failed({}): {}".format(e.error_code, e.message)
                ) from e
            cap = mvsdk.CameraGetCapability(hCamera)
            self.hCamera_list[i] = hCamera
            mono = (cap.sIspCapacity.bMonoSensor != 0)
            if mono:
                mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            mvsdk.CameraSetTriggerMode(hCamera, 0)
            mvsdk.CameraSetAeState(hCamera, 0)
            mvsdk.CameraSetExposureTime(hCamera, pg_time)
            mvsdk.CameraPlay(hCamera)
            buf_size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if mono else 3)
            pBuf = mvsdk.CameraAlignMalloc(buf_size, 16)
            self.pFrameBuffer_list[i] = pBuf
            print(f"相机 {i} 初始化完成")

    def get_frame(self):
        frames = [None, None]
        for i in range(self.nDev):
            try:
                pRaw, head = mvsdk.CameraGetImageBuffer(self.hCamera_list[i], 200)
                mvsdk.CameraImageProcess(self.hCamera_list[i], pRaw, self.pFrameBuffer_list[i], head)
                mvsdk.CameraReleaseImageBuffer(self.hCamera_list[i], pRaw)
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer_list[i], head, 1)
                data = (mvsdk.c_ubyte * head.uBytes).from_address(self.pFrameBuffer_list[i])
                frame = np.frombuffer(data, dtype=np.uint8)
                if head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8:
                    frame = frame.reshape((head.iHeight, head.iWidth, 1))
                else:
                    frame = frame.reshape((head.iHeight, head.iWidth, 3))
                frames[i] = frame
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
        self.frame_count += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.frame_count = 0
            self.last_fps_time = time.time()
        return frames

    def close_camera(self):
        for i in range(self.nDev):
            if self.hCamera_list[i]:
                mvsdk.CameraUnInit(self.hCamera_list[i])
            if self.pFrameBuffer_list[i]:
                mvsdk.CameraAlignFree(self.pFrameBuffer_list[i])

# ------------------------------- 位置阵列加载 ---------------------------------
def load_position_array(filepath: str) -> List[np.ndarray]:
    """从文本文件加载位置阵列，每行 x y z (单位：米)，支持 # 注释"""
    positions = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"位置阵列文件不存在: {filepath}")
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"警告: 第{line_num}行格式错误，跳过: {line}")
                continue
            try:
                x, y, z = map(float, parts)
                positions.append(np.array([x, y, z]))
            except ValueError:
                print(f"警告: 第{line_num}行数值解析失败，跳过: {line}")
    if not positions:
        raise ValueError("位置阵列文件中没有有效的坐标数据")
    return positions

# ------------------------------- 数据持久化（JSON 可编辑）---------------------------------
def load_existing_data(filepath_npz):
    """
    加载已有数据，优先使用 JSON（可编辑），若无则使用 NPZ
    返回 (collected列表, next_index)
    """
    json_path = filepath_npz.replace('.npz', '.json')
    # 优先尝试 JSON
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            collected = []
            for d in data['collected']:
                collected.append({
                    'true_pos': np.array(d['true_pos']),
                    'left_pt': np.array(d['left_pt']),
                    'right_pt': np.array(d['right_pt']),
                    'triangulated': np.array(d['triangulated'])
                })
            next_index = data['next_index']
            print(f"从 JSON 加载已有数据: {len(collected)} 个点，下一个索引: {next_index}")
            return collected, next_index
        except Exception as e:
            print(f"加载 JSON 失败: {e}，尝试 NPZ...")
    # 回退到 NPZ
    if not os.path.exists(filepath_npz):
        return [], 0
    try:
        data = np.load(filepath_npz)
        required = ['true_positions', 'left_points', 'right_points', 'triangulated_positions', 'next_index']
        if not all(k in data for k in required):
            print("警告: NPZ 文件格式不完整，将重新开始。")
            return [], 0
        true_positions = data['true_positions']
        left_points = data['left_points']
        right_points = data['right_points']
        triangulated = data['triangulated_positions']
        next_index = int(data['next_index'])
        n = len(true_positions)
        collected = []
        for i in range(n):
            collected.append({
                'true_pos': true_positions[i].copy(),
                'left_pt': left_points[i].copy(),
                'right_pt': right_points[i].copy(),
                'triangulated': triangulated[i].copy()
            })
        print(f"从 NPZ 加载已有数据: {len(collected)} 个点，下一个索引: {next_index}")
        return collected, next_index
    except Exception as e:
        print(f"加载 NPZ 失败: {e}，将重新开始。")
        return [], 0

def save_data(filepath_npz, collected, next_index):
    """保存数据到 NPZ 和 JSON（可编辑）"""
    if not collected:
        print("无数据可保存")
        return
    os.makedirs(os.path.dirname(filepath_npz), exist_ok=True)
    # 保存 NPZ (备份)
    true_positions = np.array([d['true_pos'] for d in collected])
    left_points = np.array([d['left_pt'] for d in collected])
    right_points = np.array([d['right_pt'] for d in collected])
    triangulated = np.array([d['triangulated'] for d in collected])
    np.savez(filepath_npz,
             true_positions=true_positions,
             left_points=left_points,
             right_points=right_points,
             triangulated_positions=triangulated,
             next_index=next_index,
             description="Accumulated calibration data with progress")
    # 保存 JSON (可编辑)
    json_path = filepath_npz.replace('.npz', '.json')
    export_data = {
        'next_index': next_index,
        'collected': [
            {
                'true_pos': d['true_pos'].tolist(),
                'left_pt': d['left_pt'].tolist(),
                'right_pt': d['right_pt'].tolist(),
                'triangulated': d['triangulated'].tolist()
            } for d in collected
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"已保存 {len(collected)} 个数据点到 {filepath_npz} 和 {json_path}，下一个索引 {next_index}")

# ------------------------------- 主程序 ---------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("乒乓球校准数据采集程序 (自动读取位置阵列，可编辑 JSON)")
    print("=" * 60)
    print("操作说明：")
    print("  - 确保小球被左右相机同时检测到（图像上会有红色圆点）")
    print("  - 按 空格键 自动使用阵列中的下一个位置，记录当前检测点")
    print("  - 按 'r' 键重置进度（从第0个点开始，会丢失未保存的进度）")
    print("  - 按 's' 键手动保存当前数据")
    print("  - 按 'q' 键退出程序并自动保存")
    print("  - 数据同时保存为 .npz 和 .json，可编辑 .json 文件回退进度")
    print("=" * 60)

    # 隐藏 Tkinter 主窗口
    root = tk.Tk()
    root.withdraw()

    # 标定文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Reuse the single runtime calibration directory instead of maintaining
    # duplicate camera-parameter copies in multiple tools.
    pram_folder = os.path.abspath(
        os.path.join(script_dir, '..', 'camera_pram_dec', 'camera_pram')
    )

    # 位置阵列文件
    position_file = os.path.join(script_dir, 'positions.txt')
    try:
        target_positions = load_position_array(position_file)
        print(f"成功加载 {len(target_positions)} 个目标位置")
    except Exception as e:
        print(f"加载位置阵列失败: {e}")
        exit(1)

    # 累积数据保存路径
    data_dir = os.path.join(script_dir, 'data')
    data_file = os.path.join(data_dir, 'calibration_data_accumulated.npz')

    # 加载已有数据及进度
    collected, next_index = load_existing_data(data_file)
    if next_index >= len(target_positions):
        print("警告: 已有进度已超过阵列总点数，将重置进度。")
        next_index = 0
        collected = []
    total_points = len(target_positions)

    cam = camera_def()
    cam.open_camera(3 * 1000)

    try:
        rel_path = os.path.join(pram_folder, 'stereo_calibration_params.npz')
        abs_path = os.path.join(pram_folder, 'absolute_extrinsic_params_standard.npz')
        params = StereoCameraParams(rel_npz_file=rel_path, abs_npz_file=abs_path)
    except Exception as e:
        print(f"加载相机参数失败: {e}")
        cam.close_camera()
        exit(1)

    aim = aim_dec()
    processing_q = queue.Queue(maxsize=2)
    display_q = queue.Queue(maxsize=2)

    cam_proc = CameraProcessor(cam, display_q, processing_q)
    det_proc = DetectionProcessor(params, aim, processing_q, display_q)

    t_cam = threading.Thread(target=cam_proc.run, daemon=True)
    t_det = threading.Thread(target=det_proc.run, daemon=True)
    t_cam.start()
    t_det.start()

    try:
        while True:
            try:
                fd = display_q.get(timeout=0.01)
            except queue.Empty:
                fd = None

            if fd is not None:
                left = fd.left_frame.copy()
                right = fd.right_frame.copy()
                if fd.pl[0] != -1:
                    cv2.circle(left, (fd.pl[0], fd.pl[1]), 5, (0, 0, 255), -1)
                if fd.pr[0] != -1:
                    cv2.circle(right, (fd.pr[0], fd.pr[1]), 5, (0, 0, 255), -1)

                status = "Ball: DETECTED" if (fd.pl[0] != -1 and fd.pr[0] != -1) else "Ball: NOT FOUND"
                color = (0, 255, 0) if "DETECTED" in status else (0, 0, 255)
                cv2.putText(left, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(left, f"Progress: {next_index}/{total_points}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 显示下一个目标位置
                if next_index < total_points:
                    target = target_positions[next_index]
                    target_text = f"Next target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
                else:
                    target_text = "All targets done!"
                cv2.putText(left, target_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

                detection = det_proc.get_latest_detection()
                if detection is not None:
                    pos = detection['triangulated']
                    pos_text = f"3D Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                else:
                    pos_text = "3D: No detection"
                cv2.putText(left, pos_text, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.putText(left, "SPACE:record  R:reset  S:save  Q:quit", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                left = cv2.resize(left, (960, 640))
                right = cv2.resize(right, (960, 640))
                combined = np.hstack((left, right))
                cv2.imshow("Dual Camera View", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("退出程序...")
                break
            elif key == ord('s'):
                if collected:
                    save_data(data_file, collected, next_index)
                else:
                    print("暂无数据，无需保存")
            elif key == ord('r'):
                if messagebox.askyesno("重置进度", "确认要重置进度吗？当前已采集的数据将保留，但下一次记录将从第0个点开始。"):
                    next_index = 0
                    print("进度已重置为0")
            elif key == ord(' '):
                if next_index >= total_points:
                    print("已完成所有目标位置的采集！按 q 退出。")
                    continue
                detection = det_proc.get_latest_detection()
                if detection is None:
                    print("⚠️ 未检测到小球，无法记录！")
                    continue

                true_pos = target_positions[next_index]
                print(f"\n当前观测位置 (三角化): x={detection['triangulated'][0]:.3f}, "
                      f"y={detection['triangulated'][1]:.3f}, z={detection['triangulated'][2]:.3f}")
                print(f"目标真实位置 #{next_index+1}: ({true_pos[0]:.3f}, {true_pos[1]:.3f}, {true_pos[2]:.3f})")

                confirm = messagebox.askyesno(
                    "确认记录",
                    f"是否记录当前点？\n真实坐标: {true_pos[0]:.3f}, {true_pos[1]:.3f}, {true_pos[2]:.3f}\n\n按是记录，按否跳过。"
                )
                if not confirm:
                    print("跳过此点")
                    continue

                collected.append({
                    'true_pos': true_pos.copy(),
                    'left_pt': np.array(detection['pl'], dtype=np.float32),
                    'right_pt': np.array(detection['pr'], dtype=np.float32),
                    'triangulated': detection['triangulated'].copy()
                })
                next_index += 1
                print(f"✅ 已记录第 {next_index} 个点 (目标 {next_index}/{total_points})")
                # 每次记录后自动保存（可选，这里注释掉了，改为手动或退出时保存）
                # save_data(data_file, collected, next_index)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cam_proc.stop()
        det_proc.stop()
        t_cam.join(timeout=1.0)
        t_det.join(timeout=1.0)

        if collected:
            save_data(data_file, collected, next_index)
        else:
            print("\n未采集任何数据")

        cam.close_camera()
        cv2.destroyAllWindows()
        root.destroy()
        print("程序结束")
