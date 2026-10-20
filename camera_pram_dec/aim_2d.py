import cv2
import numpy as np
from scipy.spatial import KDTree  # 用于快速最近邻搜索

class aim_dec(object):
    def __init__(self, min_size=15, merge_distance=50):
        """
        初始化目标检测器

        参数:
            min_size: 最小目标尺寸阈值
            merge_distance: 轮廓合并距离阈值
        """
        self.min_size = min_size
        self.merge_distance = merge_distance
        # 黄色HSV范围
        self.lower_yellow = np.array([1, 130, 50])
        self.upper_yellow = np.array([20, 230, 150])

        # 预分配内存用于常用数组
        self._temp_contours = []

    def merge_contours(self, contours):
        """
        使用空间聚类合并相近的轮廓

        参数:
            contours: 轮廓列表，每个元素为[x, y, w, h, cx, cy]

        返回:
            合并后的目标信息 [x, y, w, h, cx, cy]
        """
        if not contours:
            return [0, 0, 0, 0, -1, -1]

        # 如果只有一个轮廓，直接返回
        if len(contours) == 1:
            return contours[0]

        # 方法1: 使用质心聚类快速合并
        # 提取所有质心
        centroids = np.array([c[4:6] for c in contours], dtype=np.float32)

        # 使用KDTree进行空间聚类
        if len(centroids) > 1:
            tree = KDTree(centroids)

            # 查找距离小于阈值的点对
            pairs = tree.query_pairs(self.merge_distance)

            # 构建连通分量
            visited = [False] * len(centroids)
            clusters = []

            for i in range(len(centroids)):
                if not visited[i]:
                    cluster = [i]
                    visited[i] = True
                    stack = [i]

                    while stack:
                        current = stack.pop()
                        # 查找当前点的邻居
                        neighbors = tree.query_ball_point(centroids[current], self.merge_distance)
                        for neighbor in neighbors:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                cluster.append(neighbor)
                                stack.append(neighbor)

                    clusters.append(cluster)

            # 选择最大的簇
            if clusters:
                largest_cluster = max(clusters, key=len)

                # 计算合并后的边界框
                cluster_contours = [contours[i] for i in largest_cluster]
                x_min = min(c[0] for c in cluster_contours)
                y_min = min(c[1] for c in cluster_contours)
                x_max = max(c[2] for c in cluster_contours)
                y_max = max(c[3] for c in cluster_contours)

                # 计算合并后的质心
                cx_sum = sum(c[4] for c in cluster_contours)
                cy_sum = sum(c[5] for c in cluster_contours)
                cx = int(cx_sum / len(cluster_contours))
                cy = int(cy_sum / len(cluster_contours))

                return [x_min, y_min, x_max, y_max, cx, cy]

        # 方法2: 如果聚类失败，使用密度最高的区域
        # 计算质心的密度
        h, w = 480, 640  # 可以根据实际图像调整
        grid_h, grid_w = h // 10, w // 10  # 网格大小

        density = np.zeros((10, 10), dtype=int)
        for cx, cy in centroids:
            grid_x = min(int(cx / grid_w), 9)
            grid_y = min(int(cy / grid_h), 9)
            density[grid_y, grid_x] += 1

        # 找到密度最高的网格
        max_density_idx = np.unravel_index(np.argmax(density), density.shape)

        # 收集该网格内的轮廓
        grid_x_min = max_density_idx[1] * grid_w
        grid_y_min = max_density_idx[0] * grid_h

        in_grid = []
        for i, (cx, cy) in enumerate(centroids):
            if grid_x_min <= cx < grid_x_min + grid_w and grid_y_min <= cy < grid_y_min + grid_h:
                in_grid.append(i)

        if in_grid:
            # 计算该网格内轮廓的合并边界框
            grid_contours = [contours[i] for i in in_grid]
            x_min = min(c[0] for c in grid_contours)
            y_min = min(c[1] for c in grid_contours)
            x_max = max(c[2] for c in grid_contours)
            y_max = max(c[3] for c in grid_contours)

            # 计算合并质心
            cx_sum = sum(c[4] for c in grid_contours)
            cy_sum = sum(c[5] for c in grid_contours)
            cx = int(cx_sum / len(grid_contours))
            cy = int(cy_sum / len(grid_contours))

            return [x_min, y_min, x_max, y_max, cx, cy]

        # 如果上述方法都失败，返回最大的轮廓
        largest_contour = max(contours, key=lambda c: (c[2]-c[0])*(c[3]-c[1]))
        return largest_contour

    def filter_contours_fast(self, contours):
        """
        快速过滤和合并轮廓

        参数:
            contours: OpenCV轮廓列表

        返回:
            过滤后的轮廓信息列表
        """
        filtered = []

        # 使用NumPy向量化操作提高效率
        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 快速大小过滤
            if w >= self.min_size and h >= self.min_size:
                # 计算轮廓面积作为权重
                area = cv2.contourArea(contour)
                if area > 0:
                    # 计算质心
                    moment = cv2.moments(contour)
                    if moment['m00'] != 0:
                        cx = int(moment['m10'] / moment['m00'])
                        cy = int(moment['m01'] / moment['m00'])
                        # 存储 [x, y, w, h, area, cx, cy]
                        filtered.append([x, y, w, h, area, cx, cy])

        return filtered

    def yellow_dec_optimized(self, frame):
        """
        优化版的黄色目标检测

        参数:
            frame: 输入BGR图像

        返回:
            目标中心坐标 (cx, cy)，未检测到返回 (-1, -1)
        """
        # 1. 颜色阈值分割
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # 2. 形态学操作去噪（可选）
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (-1, -1)

        # 4. 快速过滤轮廓
        filtered_contours = self.filter_contours_fast(contours)

        if not filtered_contours:
            return (-1, -1)

        # 5. 基于面积加权的质心计算
        # 将列表转换为NumPy数组以提高效率
        filtered_array = np.array(filtered_contours)

        # 提取坐标和面积
        centroids = filtered_array[:, 5:7]  # cx, cy
        areas = filtered_array[:, 4]        # 面积

        # 计算加权质心
        total_area = np.sum(areas)
        if total_area > 0:
            weighted_cx = int(np.sum(centroids[:, 0] * areas) / total_area)
            weighted_cy = int(np.sum(centroids[:, 1] * areas) / total_area)

            # 可选：计算所有质心的中位数（对离群点更鲁棒）
            median_cx = int(np.median(centroids[:, 0]))
            median_cy = int(np.median(centroids[:, 1]))

            # 使用加权质心和中位数的平均值
            cx = int((weighted_cx + median_cx) / 2)
            cy = int((weighted_cy + median_cy) / 2)

            return (cx, cy)

        return (-1, -1)

    def yellow_dec_simple(self, frame):
        """
        简单快速版黄色目标检测
        适用于对实时性要求极高的场景
        """
        # 1. 颜色阈值分割
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # 2. 查找非零像素的坐标
        points = cv2.findNonZero(mask)

        if points is None:
            return (-1, -1)

        # 3. 计算所有黄色像素的平均位置
        points = points.reshape(-1, 2)

        # 4. 使用均值漂移或简单统计
        # 方法A：直接计算均值（最快）
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))

        # 方法B：使用中位数（对离群点更鲁棒）
        # cx = int(np.median(points[:, 0]))
        # cy = int(np.median(points[:, 1]))

        return (cx, cy)

    def yellow_dec_adaptive(self, frame, method='balanced'):
        """
        自适应选择检测方法

        参数:
            frame: 输入图像
            method: 检测方法 ('fast', 'accurate', 'balanced')

        返回:
            目标中心坐标
        """
        if method == 'fast':
            return self.yellow_dec_simple(frame)
        elif method == 'accurate':
            return self.yellow_dec(frame)
        else:  # balanced
            # 根据图像大小选择方法
            h, w = frame.shape[:2]
            if h * w > 1000000:  # 大图像使用快速方法
                return self.yellow_dec_simple(frame)
            else:
                return self.yellow_dec_optimized(frame)

    def yellow_dec(self, frame):
        """
        保持原有接口，内部使用优化方法
        """
        return self.yellow_dec_optimized(frame)

    # 可视化函数（用于调试）
    def visualize_detection(self, frame, result, method="contours"):
        """
        可视化检测结果

        参数:
            frame: 原始图像
            result: 检测结果 (cx, cy)
            method: 可视化方法
        """
        display_frame = frame.copy()
        cx, cy = result

        if cx != -1 and cy != -1:
            # 绘制目标中心
            cv2.circle(display_frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(display_frame, (cx, cy), 12, (0, 255, 0), 2)

            # 绘制十字线
            cv2.line(display_frame, (cx-20, cy), (cx+20, cy), (0, 255, 0), 2)
            cv2.line(display_frame, (cx, cy-20), (cx, cy+20), (0, 255, 0), 2)

        return display_frame


# 测试函数
def test_performance():
    """性能测试"""
    import time

    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 添加一些黄色区域
    test_image[100:150, 200:250] = [0, 200, 200]  # BGR中的黄色

    detector = aim_dec()

    # 测试不同方法的性能
    methods = ['original', 'optimized', 'simple']
    results = {}

    for method_name in methods:
        times = []
        for _ in range(100):  # 运行100次取平均
            start = time.time()

            if method_name == 'original':
                # 需要还原原始方法进行测试
                pass
            elif method_name == 'optimized':
                result = detector.yellow_dec_optimized(test_image)
            elif method_name == 'simple':
                result = detector.yellow_dec_simple(test_image)

            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times) * 1000  # 转换为毫秒
        results[method_name] = avg_time
        print(f"{method_name}: {avg_time:.2f} ms per frame")

    return results


if __name__ == '__main__':
    # 测试性能
    # results = test_performance()
    # print(f"性能对比: {results}")

    # 实际图像测试
    image = cv2.imread('./00.png')
    if image is None:
        # 如果没有测试图像，创建一个
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加黄色目标
        cv2.rectangle(image, (200, 100), (300, 200), (0, 200, 200), -1)
        cv2.rectangle(image, (350, 150), (400, 250), (0, 180, 180), -1)

    aim = aim_dec()

    # 使用优化方法
    point = aim.yellow_dec_optimized(image)
    print(f"优化方法检测结果: {point}")

    # 可视化结果
    if point[0] != -1:
        vis_image = aim.visualize_detection(image, point)
        cv2.imshow('Detection Result', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
