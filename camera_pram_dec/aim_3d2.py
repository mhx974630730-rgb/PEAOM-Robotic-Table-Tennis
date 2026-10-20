#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time
from aim_2d import aim_dec
import os

class camera_def(object):
    def __init__(self):
        # 枚举相机
        self.DevList = mvsdk.CameraEnumerateDevice()
        self.nDev = len(self.DevList)
        if self.nDev < 2:
            raise RuntimeError(f"Two cameras are required, but {self.nDev} were found")

    def open_camera(self, pg_time):
        DevInfo_list = [None, None]

        for i, DevInfo in enumerate(self.DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

        # 根据IP地址识别左右相机
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
                "port identifiers in camera_pram_dec/aim_3d2.py."
            )
        self.nDev = 2

        print(f"找到 {self.nDev} 个相机")

        # 打开相机
        self.hCamera_list = [0, 0]
        self.pFrameBuffer_list = [0, 0]

        for i, device_info in enumerate(DevInfo_list):
            hCamera = 0
            try:
                hCamera = mvsdk.CameraInit(device_info, -1, -1)
            except mvsdk.CameraException as e:
                raise RuntimeError(
                    "CameraInit failed({}): {}".format(e.error_code, e.message)
                ) from e

            # 获取相机特性描述
            cap = mvsdk.CameraGetCapability(hCamera)
            self.hCamera_list[i] = hCamera

            # 判断是黑白相机还是彩色相机
            monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

            # 设置输出格式
            if monoCamera:
                mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            # 设置连续采集模式
            mvsdk.CameraSetTriggerMode(hCamera, 0)

            # 设置手动曝光
            mvsdk.CameraSetAeState(hCamera, 0)
            mvsdk.CameraSetExposureTime(hCamera, pg_time)

            # 开始采集
            mvsdk.CameraPlay(hCamera)

            # 分配图像缓冲区
            FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
            pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
            self.pFrameBuffer_list[i] = pFrameBuffer

            print(f"相机 {i} 初始化完成")
            # 打开相机后，获取一帧并打印分辨率
            frame_test = self.get_frame()
            if frame_test[0] is not None and frame_test[1] is not None:
                print(f"左相机分辨率: {frame_test[0].shape[1]} x {frame_test[0].shape[0]}  (宽度 x 高度)")
                print(f"右相机分辨率: {frame_test[1].shape[1]} x {frame_test[1].shape[0]}")
            else:
                print("获取一帧失败，无法查看分辨率")

    def get_frame(self):
        frame_list = [None, None]
        for i in range(self.nDev):
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera_list[i], 200)
                mvsdk.CameraImageProcess(self.hCamera_list[i], pRawData, self.pFrameBuffer_list[i], FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera_list[i], pRawData)

                # Windows下图像需要翻转
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer_list[i], FrameHead, 1)

                # 转换为numpy数组
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer_list[i])
                frame = np.frombuffer(frame_data, dtype=np.uint8)

                # 根据图像类型确定通道数
                if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8:
                    frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1))
                else:
                    frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 3))

                frame_list[i] = frame
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

        return frame_list

    def close_camera(self):
        for i in range(self.nDev):
            if self.hCamera_list[i]:
                mvsdk.CameraUnInit(self.hCamera_list[i])
            if self.pFrameBuffer_list[i]:
                mvsdk.CameraAlignFree(self.pFrameBuffer_list[i])

    def show_cam(self, frame_list, pl, pr):
        """显示相机图像并标记检测点"""
        if pl[0] != -1 and pr[0] != -1:
            for i in range(len(frame_list)):
                if i == 0:
                    cv2.circle(frame_list[i], (pl[0], pl[1]), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(frame_list[i], (pr[0], pr[1]), 5, (0, 0, 255), -1)

        # 调整图像大小并拼接显示
        frame1 = cv2.resize(frame_list[0], (960, 640), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame_list[1], (960, 640), interpolation=cv2.INTER_LINEAR)
        frame_camera = np.hstack((frame1, frame2))
        cv2.imshow("Dual Camera View", frame_camera)

    def point_cau(self, target_3d_point):
        """
        将3D点坐标转换为实际坐标系（如果需要）

        注意：这个转换系数需要根据你的实际安装位置进行校准
        这里假设输入是世界坐标系下的点（米）
        """
        # 如果您的校准系数是针对世界坐标系的，可以直接使用
        # 否则可能需要调整

        # 将米转换为毫米，然后应用校准
        x_mm = target_3d_point[0] * 1000
        y_mm = target_3d_point[1] * 1000
        z_mm = target_3d_point[2] * 1000

        # 应用您的校准系数（这些系数可能需要重新校准）
        x = (x_mm - 5.2) * 30
        y = (y_mm + 18.6) * 30
        z = -(z_mm - 3.57) * 30

        return np.array([x, y, z])

class StereoCameraParams:
    """从npz文件读取双目相机参数（标准版）"""
    def __init__(self, rel_npz_file='stereo_calibration_params.npz',
                 abs_npz_file='absolute_extrinsic_params_standard.npz'):
        """
        初始化并加载相对外参和绝对外参（标准版）

        Args:
            rel_npz_file: 相对外参文件路径
            abs_npz_file: 绝对外参文件路径（标准版）
        """
        # 加载相对外参文件（双目相机之间的变换）
        rel_data = np.load(rel_npz_file)

        # 内参矩阵
        self.mtx_left = rel_data['mtx_left']  # 左相机内参
        self.mtx_right = rel_data['mtx_right']  # 右相机内参

        # 畸变系数
        self.dist_left = rel_data['dist_left']  # 左相机畸变
        self.dist_right = rel_data['dist_right']  # 右相机畸变

        # 相对外参：左右相机之间的旋转矩阵R和平移向量T
        self.R_rel = rel_data['R']  # 旋转矩阵 (3x3)
        self.T_rel = rel_data['T']  # 平移向量 (3x1)，单位：米

        # 图像尺寸
        self.image_size = tuple(rel_data['image_size'])  # (width, height)

        # 加载绝对外参文件（标准版，单位：米）
        abs_data = np.load(abs_npz_file)

        # 检查单位
        if 'units' in abs_data and abs_data['units'] == 'meters':
            print("✓ 加载标准版绝对外参（单位：米）")
        else:
            print("⚠️ 警告：外参文件可能不是标准版，确保单位一致")

        # 绝对外参：从世界到左相机的变换
        self.R_abs = abs_data['R_left']  # 从世界到左相机的旋转矩阵
        self.T_abs = abs_data['T_left']  # 从世界到左相机的平移向量，单位：米

        # 相机在世界坐标系中的位置（单位：米）
        self.cam_pos_left = abs_data['cam_pos_left']  # 米

        # 计算逆变换：从左相机到世界的变换
        # 世界 = R_inv * (相机 - T_abs)
        self.R_inv = np.linalg.inv(self.R_abs)
        self.R_world = self.R_inv  # 从左相机到世界的旋转
        self.T_world = -self.R_inv @ self.T_abs  # 从左相机到世界的平移

        # 计算投影矩阵和校正映射
        self._compute_projection_matrices()
        self._compute_rectification_maps()

        print(f"成功加载相机参数:")
        print(f"图像尺寸: {self.image_size}")
        print(f"相对基线距离: {np.linalg.norm(self.T_rel)*1000:.1f} mm")
        print(f"左相机在世界坐标系中的位置: [", self.cam_pos_left[0]*1000,self.cam_pos_left[1]*1000, self.cam_pos_left[2]*1000, "mm")

    def _compute_projection_matrices(self):
        """计算投影矩阵"""
        # 左相机投影矩阵: P_left = K_left * [I|0]
        self.P_left = np.zeros((3, 4))
        self.P_left[:, :3] = self.mtx_left

        # 右相机投影矩阵: P_right = K_right * [R_rel|T_rel]
        self.P_right = np.hstack((self.R_rel, self.T_rel))
        self.P_right = self.mtx_right @ self.P_right

    def _compute_rectification_maps(self):
        """计算立体校正映射"""
        # 计算立体校正参数
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.mtx_left, self.dist_left,
            self.mtx_right, self.dist_right,
            self.image_size, self.R_rel, self.T_rel,
            alpha=0
        )

        # 计算校正映射
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, R1, P1,
            self.image_size, cv2.CV_32FC1
        )

        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, R2, P2,
            self.image_size, cv2.CV_32FC1
        )

        self.Q = Q  # 视差到深度映射矩阵

    def undistort_and_rectify(self, left_img, right_img):
        """对左右图像进行去畸变和校正"""
        left_rect = cv2.remap(left_img, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def triangulate_point(self, left_point, right_point):
        """
        通过三角测量计算3D点坐标（左相机坐标系下）

        Args:
            left_point: 左图像中的点 (x, y)
            right_point: 右图像中的点 (x, y)

        Returns:
            左相机坐标系下的3D点坐标 (X, Y, Z)，单位：米
        """
        # 将点转换为齐次坐标
        left_pt = np.array([[left_point[0], left_point[1]]], dtype=np.float32)
        right_pt = np.array([[right_point[0], right_point[1]]], dtype=np.float32)

        # 进行三角测量
        points_4d = cv2.triangulatePoints(self.P_left, self.P_right, left_pt.T, right_pt.T)

        # 将齐次坐标转换为3D坐标
        point_3d = points_4d[:3] / points_4d[3]

        return point_3d.flatten()

    def camera_to_world(self, point_camera):
        """
        将左相机坐标系下的点转换到世界坐标系

        Args:
            point_camera: 左相机坐标系下的点 [X, Y, Z]，单位：米

        Returns:
            世界坐标系下的点 [X_w, Y_w, Z_w]，单位：米
        """
        # 应用变换：世界 = R_inv * (相机 - T_abs)
        point_world = self.R_inv @ (point_camera - self.T_abs.flatten())
        return point_world

    def triangulate_to_world(self, left_point, right_point):
        """
        直接计算世界坐标系下的坐标

        Args:
            left_point: 左图像中的点 (x, y)
            right_point: 右图像中的点 (x, y)

        Returns:
            世界坐标系下的3D点坐标 (X_w, Y_w, Z_w)，单位：米
        """
        # 1. 先得到左相机坐标系下的坐标
        point_camera = self.triangulate_point(left_point, right_point)

        # 2. 转换到世界坐标系
        point_world = self.camera_to_world(point_camera)
        point_world[2]=-point_world[2]
        return np.array([point_world[1], point_world[0], point_world[2]])

    def aim_piont_cau(self, pl, pr):
        """与原接口兼容的三角测量方法（左相机坐标系）"""
        return self.triangulate_point(pl, pr)


# camera_def 类保持不变...

if __name__ == '__main__':
    # 初始化相机
    cam_def = camera_def()
    pg_time = 3 * 1000  # 3ms曝光时间
    cam_def.open_camera(pg_time)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rel_npz_path = os.path.join(script_dir, 'camera_pram/stereo_calibration_params.npz')
    abs_npz_path = os.path.join(script_dir, 'camera_pram/absolute_extrinsic_params_standard.npz')
    # 初始化相机参数（使用相对外参和标准版绝对外参）
    try:
        cam_params = StereoCameraParams(
            rel_npz_file=rel_npz_path,
            abs_npz_file=abs_npz_path
        )
    except FileNotFoundError as e:
        print(f"错误: 找不到外参文件 - {e}")
        print("请确保以下文件存在:")
        print("1. stereo_calibration_params.npz (相对外参)")
        print("2. absolute_extrinsic_params_standard.npz (绝对外参-标准版)")
        cam_def.close_camera()
        exit(1)

    # 初始化目标检测
    aimdec = aim_dec()

    print("\n=== 定位程序已启动（标准版） ===")
    print("所有计算单位: 米")
    print("按 'q' 键退出程序")
    print("=" * 40)
    last_fps_time = time.time()
    frame_count = 0
    while True:
        # 获取图像帧
        frame_list = cam_def.get_frame()
        if frame_list[0] is None or frame_list[1] is None:
            continue

        current_time = time.time()
        # 检测目标点
        pl = aimdec.yellow_dec(frame_list[0])
        pr = aimdec.yellow_dec(frame_list[1])

        if pl[0] != -1 and pr[0] != -1:
            # 直接计算世界坐标系下的坐标
            target_point_world = cam_params.triangulate_to_world(pl, pr)
            frame_count += 1
            # 也可以获取相机坐标系下的坐标（用于调试）
            target_point_camera = cam_params.triangulate_point(pl, pr)

            # 输出结果
            print(f"\n检测到目标点:")
            print(f"相机坐标系 (米): [{target_point_camera[0]:.3f}, {target_point_camera[1]:.3f}, {target_point_camera[2]:.3f}]")
            print(f"世界坐标系 (米): [{target_point_world[0]:.3f}, {target_point_world[1]:.3f}, {target_point_world[2]:.3f}]")
            print(f"世界坐标系 (毫米): [{target_point_world[0]*1000:.1f}, {target_point_world[1]*1000:.1f}, {target_point_world[2]*1000:.1f}]")
        else:
            print('未检测到目标点')

        # 显示图像
        cam_def.show_cam(frame_list, pl, pr)
        # 计算FPS
        if current_time - last_fps_time >= 1.0:
            fps = frame_count / (current_time - last_fps_time)
            frame_count = 0
            last_fps_time = current_time
            print('FPS: '+str(fps))
        # 检查按键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理资源
    cv2.destroyAllWindows()
    cam_def.close_camera()
    print("程序结束")
