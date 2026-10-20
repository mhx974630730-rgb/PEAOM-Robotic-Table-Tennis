# generate_positions.py
import numpy as np
from pathlib import Path

def generate_grid_positions():
    # X范围 [-0.15, 0.35]，共6个点
    x = np.linspace(-0.15, 0.35, 6)
    # Y范围 [-0.4, 0.2] 步长0.1
    y = np.linspace(-0.4, 0.2, 7)
    # Z范围 [0.0175, 0.0975] 步长0.04，共3个点
    z = np.linspace(0.0175, 0.0975, 3)

    positions = []
    for zi in z:
        for yi in y:
            for xi in x:
                positions.append([xi, yi, zi])
    return positions

if __name__ == '__main__':
    positions = generate_grid_positions()
    print(f"生成 {len(positions)} 个位置")
    output_path = Path(__file__).resolve().with_name('positions.txt')
    with output_path.open('w', encoding='utf-8') as f:
        f.write("# 标定位置阵列 (x y z) 单位：米\n")
        f.write("# 范围: X=[-0.15,0.35] 6点, Y=[-0.4,0.2] 7点, "
                "Z=[0.0175,0.0975] 3点\n")
        f.write(f"# 总共 {len(positions)} 个点\n")
        for p in positions:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    print(f"已保存到 {output_path}")
