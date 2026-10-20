#!/usr/bin/env python3
# coding=utf-8

import cv2
import numpy as np
import mvsdk
import platform
import time
import threading
import queue
import os
from collections import deque
from aim_2d_test import aim_dec
from dataclasses import dataclass
from typing import Tuple
import datetime

# ========================= 数据结构 =========================
@dataclass
class FrameData:
    left_frame: np.ndarray
    right_frame: np.ndarray
    timestamp: float
    frame_id: int
    pl: Tuple[int, int] = (-1, -1)
    pr: Tuple[int, int] = (-1, -1)


# ========================= 相机采集线程 =========================
class CameraProcessor:
    def __init__(self, cam_def, display_queue=None, processing_queue=None):
        self.cam_def = cam_def
        self.display_queue = display_queue
        self.processing_queue = processing_queue
        self.running = True
        self.frame_count = 0
        self.last_fps_time = time.time()

    def run(self):
        while self.running:
            try:
                frame_list = self.cam_def.get_frame()
                if frame_list[0] is None or frame_list[1] is None:
                    time.sleep(0.001)
                    continue

                # 高精度时间戳
                current_time = time.perf_counter()

                frame_data = FrameData(
                    left_frame=frame_list[0],
                    right_frame=frame_list[1],
                    timestamp=current_time,
                    frame_id=self.frame_count
                )
                self.frame_count += 1

                # 发送到处理队列
                if self.processing_queue is not None:
                    try:
                        self.processing_queue.put(frame_data, block=False)
                    except queue.Full:
                        try:
                            self.processing_queue.get_nowait()
                            self.processing_queue.put(frame_data, block=False)
                        except queue.Empty:
                            pass

                # 发送到显示队列
                if self.display_queue is not None:
                    try:
                        self.display_queue.put(frame_data, block=False)
                    except queue.Full:
                        try:
                            self.display_queue.get_nowait()
                            self.display_queue.put(frame_data, block=False)
                        except queue.Empty:
                            pass

                # 打印 FPS
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    fps = self.frame_count / (now - self.last_fps_time)
                    print(f"相机 FPS: {fps:.1f}")
                    self.frame_count = 0
                    self.last_fps_time = now

            except Exception as e:
                print(f"相机处理错误: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False


# ========================= 检测线程 =========================
class DetectionProcessor:
    def __init__(self, cam_params, aimdec, processing_queue, display_queue=None):
        self.cam_params = cam_params
        self.aimdec = aimdec
        self.processing_queue = processing_queue
        self.display_queue = display_queue
        self.running = True
        self.detection_count = 0
        self.total_frames = 0
        self.start_time = time.time()
        self.last_status_time = time.time()

        # 轨迹记录
        self.record_mode = False
        self.record_buffer = []
        self.record_lock = threading.Lock()

    def start_recording(self):
        with self.record_lock:
            self.record_buffer.clear()
            self.record_mode = True
        print("[记录] 开始记录轨迹...")

    def stop_recording(self):
        with self.record_lock:
            self.record_mode = False
            buffer_copy = self.record_buffer.copy()
        print(f"[记录] 停止记录，共采集 {len(buffer_copy)} 个点")
        return buffer_copy

    def run(self):
        while self.running:
            try:
                frame_data = self.processing_queue.get(timeout=0.001)
            except queue.Empty:
                time.sleep(0.001)
                continue

            self.total_frames += 1

            # 目标检测
            pl = self.aimdec.yellow_dec(frame_data.left_frame)
            pr = self.aimdec.yellow_dec(frame_data.right_frame)
            frame_data.pl = pl
            frame_data.pr = pr

            if pl[0] != -1 and pr[0] != -1:
                self.detection_count += 1
                # 三角化得到世界坐标
                target_point_world = self.cam_params.triangulate_to_world(pl, pr)

                # 记录轨迹
                if self.record_mode:
                    with self.record_lock:
                        self.record_buffer.append((
                            frame_data.timestamp,
                            target_point_world.copy(),
                            pl,
                            pr
                        ))

            # 发送到显示队列
            if self.display_queue is not None:
                try:
                    self.display_queue.put(frame_data, block=False)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put(frame_data, block=False)
                    except queue.Empty:
                        pass

            # 状态输出
            now = time.time()
            if now - self.last_status_time >= 3.0:
                elapsed = now - self.start_time
                avg_fps = self.total_frames / elapsed if elapsed > 0 else 0
                status = f"检测 FPS: {avg_fps:.1f} | 检测率: {self.detection_count}/{self.total_frames}"
                if self.record_mode:
                    with self.record_lock:
                        buf_len = len(self.record_buffer)
                    status += f" | 记录中: {buf_len}点"
                print(status)
                self.last_status_time = now

    def stop(self):
        self.running = False


# ========================= 双目相机参数类 =========================
class StereoCameraParams:
    def __init__(self, rel_npz_file, abs_npz_file):
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
        print(f"成功加载相机参数: 图像尺寸 {self.image_size}, 基线距离 {np.linalg.norm(self.T_rel)*1000:.1f} mm")

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


# ========================= 相机驱动（根据实际IP修改） =========================
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
            # 请根据实际相机IP修改
            if b'197.168.58.224' in port_type:
                DevInfo_list[0] = self.DevList[i]
            elif b'198.168.58.82' in port_type:
                DevInfo_list[1] = self.DevList[i]
            else:
                print(f"未知相机端口类型: {port_type}")
        if any(device is None for device in DevInfo_list):
            raise RuntimeError(
                "Could not identify both configured cameras. Update the two "
                "port identifiers in sys_pram_dec/sys_pram_tra.py."
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
        return frames

    def close_camera(self):
        for i in range(self.nDev):
            if self.hCamera_list[i]:
                mvsdk.CameraUnInit(self.hCamera_list[i])
            if self.pFrameBuffer_list[i]:
                mvsdk.CameraAlignFree(self.pFrameBuffer_list[i])


# ========================= 主程序 =========================
if __name__ == '__main__':
    print("=" * 60)
    print("物理参数校准 - 轨迹采集程序（主线程显示，修复时间戳）")
    print("=" * 60)
    print("操作说明：")
    print("  - 在图像窗口中按 'r' 开始/停止记录轨迹")
    print("  - 按 'q' 退出程序")
    print("  - 记录停止后自动保存为 trajectory_日期时间.npz")
    print("=" * 60)

    # 初始化相机
    cam = camera_def()
    cam.open_camera(3 * 1000)  # 曝光 3ms

    # 标定文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Reuse the single runtime calibration directory instead of maintaining
    # duplicate camera-parameter copies in multiple tools.
    pram_folder = os.path.abspath(
        os.path.join(script_dir, '..', 'camera_pram_dec', 'camera_pram')
    )
    tra_folder = os.path.join(script_dir, 'tra_folder')
    os.makedirs(tra_folder, exist_ok=True)

    rel_path = os.path.join(pram_folder, 'stereo_calibration_params.npz')
    abs_path = os.path.join(pram_folder, 'absolute_extrinsic_params_standard.npz')
    try:
        params = StereoCameraParams(rel_path, abs_path)
    except Exception as e:
        print(f"加载相机参数失败: {e}")
        cam.close_camera()
        exit(1)

    aim = aim_dec()

    # 创建队列
    processing_queue = queue.Queue(maxsize=2)
    display_queue = queue.Queue(maxsize=1)

    # 创建处理器
    cam_proc = CameraProcessor(cam, display_queue, processing_queue)
    det_proc = DetectionProcessor(params, aim, processing_queue, display_queue)

    # 启动线程（相机和检测）
    t_cam = threading.Thread(target=cam_proc.run, daemon=True)
    t_det = threading.Thread(target=det_proc.run, daemon=True)
    t_cam.start()
    t_det.start()

    # 主循环：显示 + 按键响应
    try:
        while True:
            # 从显示队列获取帧（非阻塞）
            try:
                fd = display_queue.get(timeout=0.01)
            except queue.Empty:
                fd = None

            if fd is not None:
                # 准备显示图像
                left = fd.left_frame.copy()
                right = fd.right_frame.copy()
                if fd.pl[0] != -1:
                    cv2.circle(left, (fd.pl[0], fd.pl[1]), 5, (0, 0, 255), -1)
                if fd.pr[0] != -1:
                    cv2.circle(right, (fd.pr[0], fd.pr[1]), 5, (0, 0, 255), -1)

                status = "Ball: DETECTED" if (fd.pl[0] != -1 and fd.pr[0] != -1) else "Ball: NOT FOUND"
                color = (0, 255, 0) if "DETECTED" in status else (0, 0, 255)
                cv2.putText(left, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(left, "Press R to start/stop recording, Q to quit", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 调整大小并拼接
                left_resized = cv2.resize(left, (960, 640), interpolation=cv2.INTER_LINEAR)
                right_resized = cv2.resize(right, (960, 640), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack((left_resized, right_resized))
                cv2.imshow("Dual Camera View", combined)

            # 按键处理（在图像窗口中）
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("收到退出指令...")
                break
            elif key == ord('r'):
                if not det_proc.record_mode:
                    det_proc.start_recording()
                else:
                    # 停止记录并保存
                    trajectory = det_proc.stop_recording()
                    if len(trajectory) > 0:
                        timestamps = np.array([d[0] for d in trajectory])
                        positions = np.array([d[1] for d in trajectory])
                        left_pts = np.array([d[2] for d in trajectory])
                        right_pts = np.array([d[3] for d in trajectory])
                        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"trajectory_{timestamp_str}.npz"
                        save_path = os.path.join(tra_folder, filename)
                        np.savez(save_path,
                                 timestamps=timestamps,
                                 positions=positions,
                                 left_points=left_pts,
                                 right_points=right_pts,
                                 description="Trajectory data with real timestamps")
                        # 新增保存 JSON
                        json_path = save_path.replace('.npz', '.json')
                        import json
                        json_data = {
                            'timestamps': timestamps.tolist(),
                            'positions': positions.tolist(),
                            'left_points': left_pts.tolist(),
                            'right_points': right_pts.tolist()
                        }
                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        print(f"已保存轨迹 NPZ 和 JSON 到 {save_path} 和 {json_path} (共 {len(timestamps)} 点)")
                        print(f"已保存轨迹到 {save_path} (共 {len(timestamps)} 点)")
                    else:
                        print("未采集到任何点，不保存文件")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 如果还在记录，先停止并保存
        if det_proc.record_mode:
            trajectory = det_proc.stop_recording()
            if len(trajectory) > 0:
                timestamps = np.array([d[0] for d in trajectory])
                positions = np.array([d[1] for d in trajectory])
                left_pts = np.array([d[2] for d in trajectory])
                right_pts = np.array([d[3] for d in trajectory])
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trajectory_{timestamp_str}.npz"
                save_path = os.path.join(tra_folder, filename)
                np.savez(save_path,
                         timestamps=timestamps,
                         positions=positions,
                         left_points=left_pts,
                         right_points=right_pts)
                print(f"退出前已保存轨迹到 {save_path}")

        # 停止线程
        cam_proc.stop()
        det_proc.stop()
        t_cam.join(timeout=1.0)
        t_det.join(timeout=1.0)

        cam.close_camera()
        cv2.destroyAllWindows()
        print("程序结束")
