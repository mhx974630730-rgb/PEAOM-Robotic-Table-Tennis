# stereo_calibration_fixed_v2.py
import cv2
import numpy as np
import glob
import json
import os

class StereoCalibration:
    def __init__(self, chessboard_size=(7, 5), square_size=30.0):
        """
        双目标定类 - 修正版

        重要: 确保square_size的单位与实际棋盘格尺寸一致（毫米）
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size / 1000.0  # 转换为米，OpenCV内部使用米为单位
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def calibrate(self, left_dir='left', right_dir='right'):
        print("=== 双目标定开始 ===")
        print(f"棋盘格内角点: {self.chessboard_size}")
        print(f"方格实际尺寸: {self.square_size*1000:.1f} mm")
        print(f"OpenCV内部单位: 米")

        # 获取图像文件列表（确保按文件名排序）
        left_dir_t = os.path.join(self.script_dir, left_dir)
        right_dir_t = os.path.join(self.script_dir, right_dir)
        left_files = sorted(glob.glob(os.path.join(left_dir_t, '*.png')))
        right_files = sorted(glob.glob(os.path.join(right_dir_t, '*.png')))

        if len(left_files) != len(right_files):
            print(f"警告: 左右图像数量不同 (左: {len(left_files)}, 右: {len(right_files)})")
            print("将尝试按文件名配对...")

        # 创建世界坐标点（棋盘格平面Z=0）
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # 单位：米

        print(f"\n世界坐标点示例（前5个，单位：米）:")
        for i in range(min(5, len(objp))):
            print(f"  点{i}: {objp[i]}")

        # 存储数据
        objpoints = []    # 3D点
        imgpoints_left = []  # 左图像2D点
        imgpoints_right = [] # 右图像2D点

        # 处理图像对
        min_pairs = min(len(left_files), len(right_files))
        for i in range(min_pairs):
            left_path = left_files[i]
            right_path = right_files[i]

            print(f"\n处理图像对 {i+1}/{min_pairs}:")
            print(f"  左: {os.path.basename(left_path)}")
            print(f"  右: {os.path.basename(right_path)}")

            # 读取图像
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            if img_left is None or img_right is None:
                print("  错误: 无法读取图像")
                continue

            # 获取图像尺寸
            if i == 0:
                self.image_size = img_left.shape[1], img_left.shape[0]  # (width, height)
                print(f"  图像尺寸: {self.image_size}")

            # 检测角点
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)

            if ret_left and ret_right:
                # 亚像素精确化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                # 存储数据
                objpoints.append(objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)

                print("  ✓ 角点检测成功")

                # 可视化（可选）
                cv2.drawChessboardCorners(img_left, self.chessboard_size, corners_left, ret_left)
                cv2.drawChessboardCorners(img_right, self.chessboard_size, corners_right, ret_right)

                # 显示
                combined = np.hstack([img_left, img_right])
                cv2.imshow(f'Pair {i+1}', combined)
                cv2.waitKey(500)

            else:
                print(f"  ✗ 角点检测失败 (左: {ret_left}, 右: {ret_right})")

        # cv2.destroyAllWindows()

        print(f"\n总共处理了 {len(objpoints)} 对有效图像")

        if len(objpoints) < 5:
            raise ValueError(f"有效图像对不足，当前有 {len(objpoints)} 对，至少需要5对")

        # 单目标定（获取内参初值）
        print("\n1. 单目标定左相机...")
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, self.image_size, None, None)

        print("2. 单目标定右相机...")
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, self.image_size, None, None)

        print(f"左相机重投影误差: {ret_left:.6f}")
        print(f"右相机重投影误差: {ret_right:.6f}")

        # 双目标定
        print("\n3. 进行双目标定...")
        flags = cv2.CALIB_FIX_INTRINSIC  # 使用已标定的内参
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        retval, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            self.image_size, criteria=criteria, flags=flags
        )

        # 验证结果
        baseline_mm = np.linalg.norm(T) * 1000  # 转换为毫米

        print(f"\n=== 标定结果 ===")
        print(f"双目重投影误差: {retval:.6f}")
        print(f"基线距离: {baseline_mm:.1f} mm")

        # 检查基线是否合理
        if baseline_mm > 1000:  # 如果大于1米
            print(f"\n⚠️ 警告: 基线距离异常 ({baseline_mm:.1f} mm)")
            print("可能的原因:")
            print("1. 棋盘格尺寸设置错误")
            print("2. 图像未正确配对")
            print("3. 标定板移动过大")

        # 计算相机位置（假设左相机在原点）
        cam_pos_left = np.zeros(3)
        cam_pos_right = -R.T @ T  # 右相机在左相机坐标系中的位置

        print(f"\n相机相对位置:")
        print(f"左相机: (0, 0, 0) mm")
        print(f"右相机: ({cam_pos_right[0,0]*1000:.1f}, {cam_pos_right[1,0]*1000:.1f}, {cam_pos_right[2,0]*1000:.1f}) mm")

        # 构建结果
        results = {
            'intrinsics': {
                'left': {
                    'camera_matrix': mtx_left.tolist(),
                    'distortion_coeffs': dist_left.tolist(),
                    'reprojection_error': float(ret_left)
                },
                'right': {
                    'camera_matrix': mtx_right.tolist(),
                    'distortion_coeffs': dist_right.tolist(),
                    'reprojection_error': float(ret_right)
                }
            },
            'relative_extrinsics': {
                'rotation_matrix': R.tolist(),
                'translation_vector': T.tolist(),
                'baseline_mm': float(baseline_mm)
            },
            'stereo_reprojection_error': float(retval),
            'camera_positions': {
                'left_mm': [0, 0, 0],
                'right_mm': [float(cam_pos_right[0,0]*1000),
                           float(cam_pos_right[1,0]*1000),
                           float(cam_pos_right[2,0]*1000)]
            },
            'calibration_info': {
                'chessboard_size': self.chessboard_size,
                'square_size_mm': float(self.square_size * 1000),
                'image_size': self.image_size,
                'valid_image_pairs': len(objpoints),
                'units': {
                    'translation': '米',
                    'camera_positions': '毫米'
                }
            }
        }

        # 保存结果
        self._save_results(results, mtx_left, dist_left, mtx_right, dist_right, R, T)

        return results

    def _save_results(self, results, mtx_left, dist_left, mtx_right, dist_right, R, T):
        """保存标定结果"""
        # 保存为JSON
        with open('stereo_calibration_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # 保存为NPZ
        np.savez('stereo_calibration_params.npz',
                 mtx_left=mtx_left,
                 dist_left=dist_left,
                 mtx_right=mtx_right,
                 dist_right=dist_right,
                 R=R,
                 T=T,
                 image_size=results['calibration_info']['image_size'])

        print(f"\n结果已保存:")
        print(f"  - stereo_calibration_results.json")
        print(f"  - stereo_calibration_params.npz")

        # 打印关键信息
        print(f"\n关键参数:")
        print(f"左相机焦距: fx={mtx_left[0,0]:.1f}, fy={mtx_left[1,1]:.1f}")
        print(f"左相机主点: cx={mtx_left[0,2]:.1f}, cy={mtx_left[1,2]:.1f}")
        print(f"右相机焦距: fx={mtx_right[0,0]:.1f}, fy={mtx_right[1,1]:.1f}")
        print(f"右相机主点: cx={mtx_right[0,2]:.1f}, cy={mtx_right[1,2]:.1f}")
        print(f"基线: {np.linalg.norm(T)*1000:.1f} mm")
        print(f"平移向量 T (米): [{T[0,0]:.6f}, {T[1,0]:.6f}, {T[2,0]:.6f}]")


# 使用示例
if __name__ == "__main__":
    # 重要：设置正确的参数
    # chessboard_size: 棋盘格内角点数 (宽, 高)
    # square_size: 每个方格的实际尺寸（毫米）

    # 示例：如果您的棋盘格是9x6个内角点，每个方格25mm
    # CHESSBOARD_SIZE = (9, 6)
    # SQUARE_SIZE = 25.0

    # 根据您的实际情况修改
    CHESSBOARD_SIZE = (7, 5)  # 请确认这是否正确
    SQUARE_SIZE = 30.0        # 请确认方格的实际尺寸

    print("重要提示：")
    print(f"1. 棋盘格内角点: {CHESSBOARD_SIZE}")
    print(f"2. 方格实际尺寸: {SQUARE_SIZE} mm")
    print(f"3. 确保left和right目录中的图像一一对应")
    print(f"4. 确保图像文件名排序正确")

    input("按Enter键开始标定...")

    calib = StereoCalibration(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)

    try:
        results = calib.calibrate(left_dir='left', right_dir='right')

        # 验证结果
        baseline = results['relative_extrinsics']['baseline_mm']
        if baseline < 50 or baseline > 500:
            print(f"\n⚠️ 警告: 基线距离 {baseline:.1f} mm 可能不合理")
            print("典型的双目相机基线距离在50-300mm之间")
            print("请检查:")
            print("1. 棋盘格尺寸是否正确")
            print("2. 图像是否配对正确")
            print("3. 是否使用了正确的棋盘格内角点数")

    except Exception as e:
        print(f"标定失败: {e}")
        import traceback
        traceback.print_exc()
