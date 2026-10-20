import tkinter as tk
import cv2
from PIL import Image ,ImageTk

# 定义全局变量
image_path = './0_0.png'
image = None

def get_hsv_value(event):
    global image

    # 获取鼠标点击位置的坐标
    x = event.x
    y = event.y

    if image is not None:
        # 将图像从BGR颜色空间转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 获取点击位置的HSV值
        hsv_value = hsv_image[y, x]

        # 分别获取H、S、V分量的值
        hue = hsv_value[0]
        saturation = hsv_value[1]
        value = hsv_value[2]

        # 打印HSV值
        print("HSV值:H={}, S={}, V={}".format(hue, saturation, value))

def main():
    global image

    # 创建窗口
    window = tk.Tk()
    window.title("图像查看器")

    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为Tkinter可用的格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))

    # 创建画布
    canvas = tk.Canvas(window, width=image.shape[1], height=image.shape[0])
    canvas.pack()

    # 显示图像
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

    # 绑定鼠标点击事件
    canvas.bind("<Button-1>", get_hsv_value)

    # 运行窗口主循环
    window.mainloop()

if __name__ == '__main__':
    main()
