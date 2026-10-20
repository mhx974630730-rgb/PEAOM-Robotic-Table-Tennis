import cv2
import numpy as np

class aim_dec(object):
    def __init__(self):
        pass
    def calculateDistance(self,a,b):
        distance = ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        return distance

    def filterPoints(self, points, threshold,frame):
        filteredPoints = []
        if len(points) == 1:
            filteredPoint = points[0]
        else:
            for i in range(len(points)):
                currentPoint = points[i]
                k = currentPoint[0:4]
                z = currentPoint[4:6]
                if k[2]-k[0] >= threshold and k[3]-k[1]>=threshold:
                    filteredPoints.append(currentPoint)
            #for i in range(len(filteredPoints)):
                #currentPoint2 = filteredPoints[i]
                #k2 = currentPoint2[0:4]
                #x = k2[0]
                #y = k2[1]
                #w = k2[2]
                #h = k2[3]
                #cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            for i in range(len(filteredPoints)):
                currentPoint2 = filteredPoints[i]
                k2 = currentPoint2[0:4]
                if i == 0:
                    x = k2[0]
                    y = k2[1]
                    w = k2[2]
                    h = k2[3]
                else:
                    x = min(x,k2[0])
                    y = min(y,k2[1])
                    w = max(w,k2[2])
                    h = max(h,k2[3])
            if len(filteredPoints) != 0:
                cx = int((x+w)/2)
                cy = int((y+h)/2)
            else:
                x = 0
                y = 0
                w = 0
                h = 0
                cx = -1
                cy = -1
            filteredPoint = [x, y, w, h, cx, cy]
        return filteredPoint

    def yellow_dec(self, frame):
        # 将BGR图片转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 定义橙黄色的HSV范围
        lower_yellow = np.array([1, 130, 50])
        upper_yellow = np.array([20, 230, 150])

        # 根据阈值构建掩模
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓并计算轮廓的质心
        # 显示图片

        points = []
        for contour in contours:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算轮廓的质心
            moment = cv2.moments(contour)
            if moment['m00'] != 0:
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])
                points.append([x, y, w+x, h+y, cx, cy])
                #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                #cv2.rectangle(frame, (x, w+x), (y, h+y), (255, 0, 0), 2)



        if len(points) != 0:
            filter_points = self.filterPoints(points, 15, frame)
            #cv2.circle(frame, (filter_points[4], filter_points[5]), 5, (0, 0, 255), -1)
            #cv2.rectangle(frame, (filter_points[0], filter_points[1]), (filter_points[2], filter_points[3]), (255, 0, 0), 2)
            #cv2.imshow('Detected Colors', frame)

            return (filter_points[4],filter_points[5])
        else:
            #cv2.imshow('Detected Colors', frame)
            return (-1,-1)

if __name__ == '__main__':
    image = cv2.imread('./00.png')
    aim = aim_dec()
    point = aim.yellow_dec(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(point)
    # 在原图上绘制矩形和质心
