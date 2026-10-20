#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, savgol_filter

# ==================== 用户可调参数 ====================
SMOOTH_POS_SIGMA = 1.5           # 位置平滑sigma
VEL_SMOOTH_WINDOW = 9            # 速度平滑窗口（奇数）
VEL_SMOOTH_POLY = 2              # 速度平滑多项式阶数
PEAK_ORDER = 5                   # 极值检测邻域半径
COLLISION_WINDOW = 0.3           # 碰撞前后寻找峰值的时间窗口（秒）
VEL_FIT_WINDOW = 0.1             # 碰撞前用于拟合速度的时间窗口（秒，从碰撞点向前）
G_MIN = 8.5                      # 合理的 g 最小值
G_MAX = 10.5                     # 合理的 g 最大值
E_MIN = 0.7                      # 合理的 e 最小值
E_MAX = 1.05                     # 合理的 e 最大值
SHOW_DETAILS = True
SHOW_PLOTS = False
SAVE_PLOTS = False
# ===================================================

def load_trajectory(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.npz':
            data = np.load(filepath)
            t = data['timestamps']
            pos = data['positions']
        elif ext == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            t = np.array(data['timestamps'])
            pos = np.array(data['positions'])
        else:
            raise ValueError
        if pos.shape[1] >= 3:
            z = pos[:, 2]
        else:
            z = pos[:, 0] if pos.shape[1] == 1 else pos[:, 1]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None
    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t = t[idx]
        z = z[idx]
    if len(z) > 10:
        z = gaussian_filter1d(z, sigma=SMOOTH_POS_SIGMA)
    return t, z

def compute_velocity(t, z):
    dt = np.diff(t)
    v_raw = np.diff(z) / dt
    t_mid = (t[:-1] + t[1:]) / 2
    if len(v_raw) >= VEL_SMOOTH_WINDOW:
        v = savgol_filter(v_raw, VEL_SMOOTH_WINDOW, VEL_SMOOTH_POLY)
    else:
        v = v_raw
    return t_mid, v

def find_first_collision(t, z):
    min_idx = argrelextrema(z, np.less, order=PEAK_ORDER)[0]
    if len(min_idx) == 0:
        return None
    return min_idx[0]

def find_peaks_around_collision(t, z, coll_idx, window=0.3):
    t_coll = t[coll_idx]
    mask_before = (t >= t_coll - window) & (t < t_coll)
    mask_after = (t > t_coll) & (t <= t_coll + window)
    if np.sum(mask_before) < 3 or np.sum(mask_after) < 3:
        return None, None, None, None
    t_before = t[mask_before]
    z_before = z[mask_before]
    t_after = t[mask_after]
    z_after = z[mask_after]
    # 碰撞前峰值
    max_before_idx = argrelextrema(z_before, np.greater, order=PEAK_ORDER//2)[0]
    if len(max_before_idx) == 0:
        idx_before = np.argmax(z_before)
    else:
        heights = z_before[max_before_idx]
        idx_before = max_before_idx[np.argmax(heights)]
    # 碰撞后峰值
    max_after_idx = argrelextrema(z_after, np.greater, order=PEAK_ORDER//2)[0]
    if len(max_after_idx) == 0:
        idx_after = np.argmax(z_after)
    else:
        heights = z_after[max_after_idx]
        idx_after = max_after_idx[np.argmax(heights)]
    return t_before[idx_before], z_before[idx_before], t_after[idx_after], z_after[idx_after]

def fit_gravity_from_velocity(t_mid, v, t_coll, window=0.1):
    mask = (t_mid >= t_coll - window) & (t_mid < t_coll)
    t_seg = t_mid[mask]
    v_seg = v[mask]
    if len(t_seg) < 3:
        return None
    coeffs = np.polyfit(t_seg, v_seg, 1)
    g = abs(coeffs[0])
    if g < G_MIN or g > G_MAX:
        return None
    return g

def process_trajectory(filepath, window=0.3, vel_window=0.1,
                       show_details=False, show_plot=False, save_plot=False):
    t, z = load_trajectory(filepath)
    if t is None:
        return None, None
    coll_idx = find_first_collision(t, z)
    if coll_idx is None:
        print("  No collision")
        return None, None
    t_coll = t[coll_idx]
    z_coll = z[coll_idx]

    # 计算速度
    t_mid, v = compute_velocity(t, z)

    # 拟合重力加速度（速度线性回归）
    g = fit_gravity_from_velocity(t_mid, v, t_coll, window=vel_window)
    if g is None:
        print("  Gravity fit failed (velocity linear regression)")
        g = None

    # 寻找碰撞前后峰值（高度法）
    t_peak_before, z_peak_before, t_peak_after, z_peak_after = find_peaks_around_collision(t, z, coll_idx, window)
    e = None
    if t_peak_before is not None and t_peak_after is not None:
        h_before = z_peak_before - z_coll
        h_after = z_peak_after - z_coll
        if h_before > 0 and h_after > 0:
            e = np.sqrt(h_after / h_before)
            if e < E_MIN or e > E_MAX:
                e = None

    if show_details:
        print(f"  Collision: t={t_coll:.3f}, z={z_coll:.4f}")
        if t_peak_before is not None:
            print(f"  Peak before: t={t_peak_before:.3f}, z={z_peak_before:.4f}")
            print(f"  Peak after:  t={t_peak_after:.3f}, z={z_peak_after:.4f}")
            print(f"  Heights: before={h_before:.4f}, after={h_after:.4f} -> e={e:.4f}")
        if g is not None:
            print(f"  g (velocity) = {g:.3f} m/s²")

    # 绘图
    if show_plot or save_plot:
        plt.figure(figsize=(12, 8))
        plt.subplot(2,1,1)
        plt.plot(t, z, 'b-', label='Z')
        plt.scatter(t_coll, z_coll, color='r', s=50, label='Collision')
        if t_peak_before is not None:
            plt.scatter(t_peak_before, z_peak_before, color='g', s=50, label='Pre-peak')
            plt.scatter(t_peak_after, z_peak_after, color='m', s=50, label='Post-peak')
        plt.ylabel('Z (m)')
        plt.legend()
        plt.grid(True)
        plt.subplot(2,1,2)
        plt.plot(t_mid, v, 'g-', label='Velocity')
        plt.axvline(x=t_coll, color='r', linestyle='--')
        if g is not None:
            mask = (t_mid >= t_coll - vel_window) & (t_mid < t_coll)
            t_fit = t_mid[mask]
            if len(t_fit) >= 2:
                coeffs = np.polyfit(t_fit, v[mask], 1)
                t_line = np.array([t_fit[0], t_fit[-1]])
                v_line = np.polyval(coeffs, t_line)
                plt.plot(t_line, v_line, 'm--', label=f'g fit: {g:.2f}')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.grid(True)
        plt.suptitle(os.path.basename(filepath))
        if save_plot:
            plt.savefig(filepath + '_vel_fit.png')
        if show_plot:
            plt.show()
        else:
            plt.close()

    return g, e

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_tra_folder = os.path.join(script_dir, 'tra_folder')
    inputs = [default_tra_folder] if os.path.exists(default_tra_folder) else ['.']

    traj_files = []
    for path in inputs:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(('.npz', '.json')):
                    traj_files.append(os.path.join(path, f))
        elif os.path.exists(path):
            traj_files.append(path)

    if not traj_files:
        print("No trajectory files found.")
        return

    all_g = []
    all_e = []

    for f in traj_files:
        print(f"Processing: {os.path.basename(f)}")
        g, e = process_trajectory(f, window=COLLISION_WINDOW, vel_window=VEL_FIT_WINDOW,
                                  show_details=SHOW_DETAILS, show_plot=SHOW_PLOTS, save_plot=SAVE_PLOTS)
        if g is not None:
            print(f"  g = {g:.3f} m/s²")
            all_g.append(g)
        else:
            print(f"  g = N/A")
        if e is not None:
            print(f"  e = {e:.4f}")
            all_e.append(e)
        else:
            print(f"  e = N/A")

    # 剔除离群值（2σ）
    if len(all_g) > 1:
        mean_g = np.mean(all_g)
        std_g = np.std(all_g)
        all_g = [g for g in all_g if abs(g - mean_g) < 2*std_g]
    if len(all_e) > 1:
        mean_e = np.mean(all_e)
        std_e = np.std(all_e)
        all_e = [e for e in all_e if abs(e - mean_e) < 2*std_e]

    print("\n" + "="*60)
    print("Summary (Velocity-based g + Height-based e)")
    print("="*60)
    if all_g:
        mean_g = np.mean(all_g)
        std_g = np.std(all_g)
        print(f"Gravity acceleration g = {mean_g:.3f} ± {std_g:.3f} m/s²")
        print(f"Relative error to 9.81: {abs(mean_g-9.81)/9.81*100:.2f}%")
    else:
        print("No valid g measurements.")
    if all_e:
        mean_e = np.mean(all_e)
        std_e = np.std(all_e)
        print(f"Restitution coefficient e = {mean_e:.3f} ± {std_e:.3f}")
    else:
        print("No valid e measurements.")

    output_path = os.path.join(script_dir, 'physics_params_velocity_g.txt')
    with open(output_path, 'w') as f:
        f.write("Physics Parameter Fitting (Velocity g + Height e)\n")
        f.write("================================================\n")
        f.write(f"Number of files: {len(traj_files)}\n")
        if all_g:
            f.write(f"Gravity g = {mean_g:.4f} ± {std_g:.4f} m/s²\n")
        if all_e:
            f.write(f"Restitution e = {mean_e:.4f} ± {std_e:.4f}\n")
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()