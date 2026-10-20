# absolute_extrinsic_calibration_standard.py
import cv2
import numpy as np
import json
import os

class AbsoluteExtrinsicCalibration:
    def __init__(self, chessboard_size=(7, 5), square_size=30.0):
        """
        绝对外参标定 - 标准版

        重要: 与双目标定程序保持一致，使用米为单位
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size / 1000.0  # 转换为米，与双目标定一致

        # 获取脚本所在目录，后续所有相对路径都基于此
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def calibrate(self, calibration_params_path='stereo_calibration_params.npz'):
        print("=== 绝对外参标定开始（标准版） ===")
        print(f"棋盘格内角点: {self.chessboard_size}")
        print(f"方格实际尺寸: {self.square_size*1000:.1f} mm")
        print(f"内部计算单位: 米")

        # 1. 加载双目标定参数
        # 如果传入的是相对路径，则基于脚本目录转换为绝对路径
        if not os.path.isabs(calibration_params_path):
            calibration_params_path = os.path.join(self.script_dir, calibration_params_path)
        if not os.path.exists(calibration_params_path):
            raise FileNotFoundError(f"未找到参数文件: {calibration_params_path}")

        data = np.load(calibration_params_path)
        self.mtx_left = data['mtx_left']
        self.dist_left = data['dist_left']
        self.mtx_right = data['mtx_right']
        self.dist_right = data['dist_right']

        print("已加载双目标定参数")

        # 2. 查找左右图像（相对于脚本目录）
        left_dir = os.path.join(self.script_dir, 'wc', 'left')
        right_dir = os.path.join(self.script_dir, 'wc', 'right')

        if not os.path.exists(left_dir):
            raise FileNotFoundError(f"左图像目录不存在: {left_dir}")
        if not os.path.exists(right_dir):
            raise FileNotFoundError(f"右图像目录不存在: {right_dir}")

        left_images = [f for f in os.listdir(left_dir) if f.endswith('.png')]
        right_images = [f for f in os.listdir(right_dir) if f.endswith('.png')]

        if not left_images:
            raise ValueError(f"在 {left_dir} 中未找到PNG图像")
        if not right_images:
            raise ValueError(f"在 {right_dir} 中未找到PNG图像")

        # 3. 尝试找到匹配的图像
        common_names = set(left_images) & set(right_images)

        if common_names:
            image_name = sorted(common_names)[0]
            left_path = os.path.join(left_dir, image_name)
            right_path = os.path.join(right_dir, image_name)
            print(f"选择匹配的图像: {image_name}")
        else:
            left_path = os.path.join(left_dir, sorted(left_images)[0])
            right_path = os.path.join(right_dir, sorted(right_images)[0])
            print(f"左图像: {os.path.basename(left_path)}")
            print(f"右图像: {os.path.basename(right_path)}")
            print("警告：左右图像文件名不匹配")

        # 4. 读取并处理图像
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None:
            raise ValueError(f"无法读取左图像: {left_path}")
        if img_right is None:
            raise ValueError(f"无法读取右图像: {right_path}")

        # 5. 检测角点
        print("\n检测棋盘格角点...")
        ret_left, corners_left = self._find_corners(img_left)
        ret_right, corners_right = self._find_corners(img_right)

        if not ret_left:
            raise ValueError("左图像中未检测到棋盘格")
        if not ret_right:
            raise ValueError("右图像中未检测到棋盘格")

        print("两个图像都成功检测到棋盘格角点")

        # 6. 创建世界坐标点（单位：米）
        objp = self._create_object_points()

        print(f"世界坐标点示例（前5个，单位：米）:")
        for i in range(min(5, len(objp))):
            print(f"  点{i}: {objp[i]}")

        # 7. 计算绝对外参
        _, rvec_left, tvec_left = cv2.solvePnP(
            objp, corners_left, self.mtx_left, self.dist_left)

        _, rvec_right, tvec_right = cv2.solvePnP(
            objp, corners_right, self.mtx_right, self.dist_right)

        R_left, _ = cv2.Rodrigues(rvec_left)
        R_right, _ = cv2.Rodrigues(rvec_right)

        # 计算相机在世界坐标系中的位置（米）
        cam_pos_left_m = -R_left.T @ tvec_left
        cam_pos_right_m = -R_right.T @ tvec_right

        # 8. 保存结果（保存在脚本目录下）
        results = self._build_results_dict(
            R_left, rvec_left, tvec_left, cam_pos_left_m,
            R_right, rvec_right, tvec_right, cam_pos_right_m
        )

        self._print_results(results)
        self._save_results(results)

        # 9. 可视化
        self._visualize_results(
            img_left, img_right,
            corners_left, corners_right,
            rvec_left, tvec_left,
            rvec_right, tvec_right
        )

        return results

    def _create_object_points(self):
        """创建世界坐标点（单位：米）"""
        w, h = self.chessboard_size
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        objp *= self.square_size  # 单位：米
        return objp

    def _find_corners(self, image):
        """检测棋盘格角点"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners

    def _build_results_dict(self, R_left, rvec_left, tvec_left, cam_pos_left_m,
                           R_right, rvec_right, tvec_right, cam_pos_right_m):
        """构建结果字典"""
        return {
            'world_coordinate_system': {
                'description': '固定在棋盘格平面',
                'origin': '棋盘格第一个角点（左下角）',
                'x_axis': '棋盘格水平方向',
                'y_axis': '棋盘格垂直方向',
                'z_axis': '垂直于棋盘格平面（向外）',
                'units': '米(m)'
            },
            'left_camera': {
                'rotation_matrix': R_left.tolist(),
                'rotation_vector': rvec_left.tolist(),
                'translation_vector': tvec_left.tolist(),  # 单位：米
                'camera_position': cam_pos_left_m.tolist()  # 单位：米
            },
            'right_camera': {
                'rotation_matrix': R_right.tolist(),
                'rotation_vector': rvec_right.tolist(),
                'translation_vector': tvec_right.tolist(),  # 单位：米
                'camera_position': cam_pos_right_m.tolist()  # 单位：米
            },
            'calibration_info': {
                'chessboard_size': self.chessboard_size,
                'square_size_mm': float(self.square_size * 1000),
                'units': {
                    'translation_vectors': '米',
                    'camera_positions': '米'
                }
            }
        }

    def _print_results(self, results):
        """打印结果"""
        print("\n" + "="*60)
        print("绝对外参标定结果（标准版）")
        print("="*60)

        # 获取以米为单位的位置
        pos_left_m = np.array(results['left_camera']['camera_position']).flatten()
        pos_right_m = np.array(results['right_camera']['camera_position']).flatten()

        # 获取以毫米为单位的位置（用于显示）
        pos_left_mm = pos_left_m * 1000
        pos_right_mm = pos_right_m * 1000

        print(f"\n左相机位置:")
        print(f"  米: X={pos_left_m[0]:.3f}, Y={pos_left_m[1]:.3f}, Z={pos_left_m[2]:.3f}")
        print(f"  毫米: X={pos_left_mm[0]:.1f}, Y={pos_left_mm[1]:.1f}, Z={pos_left_mm[2]:.1f}")

        print(f"\n右相机位置:")
        print(f"  米: X={pos_right_m[0]:.3f}, Y={pos_right_m[1]:.3f}, Z={pos_right_m[2]:.3f}")
        print(f"  毫米: X={pos_right_mm[0]:.1f}, Y={pos_right_mm[1]:.1f}, Z={pos_right_mm[2]:.1f}")

        # 计算相机间距
        distance_m = np.linalg.norm(pos_left_m - pos_right_m)
        distance_mm = distance_m * 1000

        print(f"\n相机间距:")
        print(f"  {distance_m:.3f} 米")
        print(f"  {distance_mm:.1f} 毫米")

        # 获取平移向量
        T_left = np.array(results['left_camera']['translation_vector']).flatten()
        T_right = np.array(results['right_camera']['translation_vector']).flatten()

        print(f"\n平移向量 T_left (米): [{T_left[0]:.6f}, {T_left[1]:.6f}, {T_left[2]:.6f}]")
        print(f"平移向量 T_right (米): [{T_right[0]:.6f}, {T_right[1]:.6f}, {T_right[2]:.6f}]")

    def _save_results(self, results):
        """保存结果到文件（保存在脚本目录下）"""
        # 保存为JSON
        json_path = os.path.join(self.script_dir, 'absolute_extrinsic_results_standard.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # 保存为NPZ（单位：米）
        npz_path = os.path.join(self.script_dir, 'absolute_extrinsic_params_standard.npz')
        np.savez(npz_path,
                 R_left=np.array(results['left_camera']['rotation_matrix']),
                 T_left=np.array(results['left_camera']['translation_vector']),  # 单位：米
                 cam_pos_left=np.array(results['left_camera']['camera_position']),  # 单位：米
                 R_right=np.array(results['right_camera']['rotation_matrix']),
                 T_right=np.array(results['right_camera']['translation_vector']),  # 单位：米
                 cam_pos_right=np.array(results['right_camera']['camera_position']),  # 单位：米
                 units='meters')

        print("\n结果已保存:")
        print(f"  - {json_path}")
        print(f"  - {npz_path}")
        print(f"  单位: 米")

    def _visualize_results(self, img_left, img_right, corners_left, corners_right,
                          rvec_left, tvec_left, rvec_right, tvec_right):
        """可视化结果"""
        # 绘制角点
        cv2.drawChessboardCorners(img_left, self.chessboard_size, corners_left, True)
        cv2.drawChessboardCorners(img_right, self.chessboard_size, corners_right, True)

        # 绘制坐标轴（使用50mm长度的轴）
        axis_length = 0.05  # 米，对应50mm
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],      # X轴
            [0, axis_length, 0],      # Y轴
            [0, 0, -axis_length]      # Z轴（负值表示指向相机外）
        ]).reshape(-1, 3)

        # 左图像坐标轴
        imgpts_left, _ = cv2.projectPoints(axis_points, rvec_left, tvec_left,
                                          self.mtx_left, self.dist_left)
        origin = tuple(imgpts_left[0].ravel().astype(int))
        img_left = cv2.line(img_left, origin, tuple(imgpts_left[1].ravel().astype(int)), (0, 0, 255), 3)
        img_left = cv2.line(img_left, origin, tuple(imgpts_left[2].ravel().astype(int)), (0, 255, 0), 3)
        img_left = cv2.line(img_left, origin, tuple(imgpts_left[3].ravel().astype(int)), (255, 0, 0), 3)

        # 右图像坐标轴
        imgpts_right, _ = cv2.projectPoints(axis_points, rvec_right, tvec_right,
                                           self.mtx_right, self.dist_right)
        origin = tuple(imgpts_right[0].ravel().astype(int))
        img_right = cv2.line(img_right, origin, tuple(imgpts_right[1].ravel().astype(int)), (0, 0, 255), 3)
        img_right = cv2.line(img_right, origin, tuple(imgpts_right[2].ravel().astype(int)), (0, 255, 0), 3)
        img_right = cv2.line(img_right, origin, tuple(imgpts_right[3].ravel().astype(int)), (255, 0, 0), 3)

        # 调整图像大小并显示
        img_left_resized = cv2.resize(img_left, (960, 720))
        img_right_resized = cv2.resize(img_right, (960, 720))

        cv2.imshow('Left Camera (Red:X, Green:Y, Blue:Z)', img_left_resized)
        cv2.imshow('Right Camera (Red:X, Green:Y, Blue:Z)', img_right_resized)

        print("\n显示结果图像中...")
        print("红色: X轴, 绿色: Y轴, 蓝色: Z轴 (50mm长度)")
        print("按任意键继续...")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    print("=== 绝对外参标定程序（标准版） ===")
    print("重要提示:")
    print("1. 本程序使用米为单位，与双目标定程序保持一致")
    print("2. 世界坐标系以棋盘格第一个角点为原点")
    print("3. 请确保棋盘格放置稳定，图像清晰")

    # 参数设置（与双目标定程序相同）
    CHESSBOARD_SIZE = (7, 5)  # 请确认这是否正确
    SQUARE_SIZE = 30.0        # 方格实际尺寸（毫米）

    print(f"\n参数设置:")
    print(f"  棋盘格内角点: {CHESSBOARD_SIZE}")
    print(f"  方格实际尺寸: {SQUARE_SIZE} mm")

    input("\n按Enter键开始标定...")

    try:
        calib = AbsoluteExtrinsicCalibration(
            chessboard_size=CHESSBOARD_SIZE,
            square_size=SQUARE_SIZE
        )

        results = calib.calibrate()

        print("\n✓ 标定完成!")
        print(f"\n接下来:")
        print(f"1. 使用 'absolute_extrinsic_params_standard.npz' 文件")
        print(f"2. 修改定位程序，使用这个新文件")
        print(f"3. 确保定位程序中的单位转换代码已移除（因为现在单位统一）")

    except Exception as e:
        print(f"\n✗ 标定失败: {e}")
        import traceback
        traceback.print_exc()
