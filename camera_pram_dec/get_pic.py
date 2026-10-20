#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time
import os  # 新增

def main_loop():
    # 获取脚本所在目录，并创建保存图像的子目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    left_dir = os.path.join(script_dir, 'wc',"left")
    right_dir = os.path.join(script_dir, 'wc', "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 2:
        raise RuntimeError(f"Two cameras are required, but {nDev} were found")
    DevInfo_list = [None,None]
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    for i in range(nDev):
        if b'198.168.58.82' in DevList[i].acPortType:
            DevInfo_list[0] = DevList[i]
        elif b'197.168.58.224' in DevList[i].acPortType:
            DevInfo_list[1] = DevList[i]
        else:
            print(f"未知相机端口类型: {DevList[i].acPortType}")
    if any(device is None for device in DevInfo_list):
        raise RuntimeError(
            "Could not identify both configured cameras. Update the two "
            "port identifiers in camera_pram_dec/get_pic.py."
        )
    nDev = 2
    hCamera_list = [0,0]
    pFrameBuffer_list = [0,0]

    for i, device_info in enumerate(DevInfo_list):
        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(device_info, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
            return

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)
        hCamera_list[i] = hCamera
        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(hCamera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 20*1000)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        pFrameBuffer_list[i] = pFrameBuffer
    num_pic = 1
    for h in range(num_pic):
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            for i in range(nDev):
                # 从相机取一帧图片
                try:
                    pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera_list[i], 200)
                    mvsdk.CameraImageProcess(hCamera_list[i], pRawData, pFrameBuffer_list[i], FrameHead)
                    mvsdk.CameraReleaseImageBuffer(hCamera_list[i], pRawData)

                    # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
                    # linux下直接输出正的，不需要上下翻转
                    if platform.system() == "Windows":
                        mvsdk.CameraFlipFrameBuffer(pFrameBuffer_list[i], FrameHead, 1)

                    # 此时图片已经存储在pFrameBuffer中
                    frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer_list[i])
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

                    # 使用绝对路径保存图像
                    if i == 0:
                        save_path = os.path.join(left_dir, str(h) + '.png')
                    else:
                        save_path = os.path.join(right_dir, str(h) + '.png')
                    cv2.imwrite(save_path, frame)

                    cv2.imshow("Press q to end"+str(i)+str(h), frame)

                except mvsdk.CameraException as e:
                    if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                        print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
        cv2.destroyAllWindows()
    for i in range(nDev):
        # 关闭相机
        mvsdk.CameraUnInit(hCamera_list[i])
        # 释放帧缓存
        mvsdk.CameraAlignFree(pFrameBuffer_list[i])

def main():
    try:
        main_loop()
    finally:
        cv2.destroyAllWindows()

main()
