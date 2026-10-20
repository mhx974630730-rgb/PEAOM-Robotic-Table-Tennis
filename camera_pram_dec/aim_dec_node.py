#!/usr/bin/env python3
#coding=utf-8

import os
from aim_3d2 import camera_def, StereoCameraParams
import rospy
from geometry_msgs.msg import PointStamped
from aim_2d import aim_dec

class camera_node(object):
    def __init__(self):
        rospy.init_node('Camera_node', anonymous=True)
        self.pub = rospy.Publisher('/ball_position', PointStamped, queue_size=1)
        self.rate = rospy.Rate(90)

    def pub_ball_pose(self, ballpose):
        bp = PointStamped()
        bp.header.stamp = rospy.Time.now()          # 添加时间戳
        bp.header.frame_id = "world"                # 设置坐标系，可根据实际情况修改
        bp.point.x = ballpose[0]
        bp.point.y = ballpose[1]
        bp.point.z = ballpose[2]
        self.pub.publish(bp)
        #rospy.loginfo(bp)
        self.rate.sleep()

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cam_def = camera_def()
    pg_time = 3 * 1000
    cam_def.open_camera(pg_time)

    # 构造标定参数文件的绝对路径
    rel_npz_path = os.path.join(script_dir, 'camera_pram/stereo_calibration_params.npz')
    abs_npz_path = os.path.join(script_dir, 'camera_pram/absolute_extrinsic_params_standard.npz')

    # 初始化相机参数（使用相对外参和标准版绝对外参）
    try:
        cam_params = StereoCameraParams(
            rel_npz_file=rel_npz_path,
            abs_npz_file=abs_npz_path  # 使用标准版
        )
    except FileNotFoundError as e:
        print(f"错误: 找不到外参文件 - {e}")
        print("请确保以下文件存在于脚本所在目录:")
        print("1. stereo_calibration_params.npz (相对外参)")
        print("2. absolute_extrinsic_params_standard.npz (绝对外参-标准版)")
        cam_def.close_camera()
        exit(1)

    aimdec = aim_dec()
    cam_node = camera_node()

    try:
        while not rospy.is_shutdown():
            frame_list = cam_def.get_frame()
            pl = aimdec.yellow_dec(frame_list[0])
            pr = aimdec.yellow_dec(frame_list[1])
            if pl[0] != -1 and pr[0] != -1:
                target_point_3d = cam_params.triangulate_to_world(pl, pr)
                rospy.loginfo(
                    f"target_point =({target_point_3d[0]:.5f},"
                    f"{target_point_3d[1]:.5f},{target_point_3d[2]:.5f})"
                )
                cam_node.pub_ball_pose(target_point_3d)
            else:
                rospy.logdebug("Ball was not detected in both camera frames")
    finally:
        cam_def.close_camera()
