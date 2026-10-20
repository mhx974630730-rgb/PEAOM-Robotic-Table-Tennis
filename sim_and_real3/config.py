# config.py
import os
from experiment_config import DATA_DIR, MODEL_DIR, TRAIN_MODE

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ==================== 权重路径 ====================
SIM_WEIGHT_PATH = os.path.join(MODEL_DIR, 'sim_actor_weights.pth')
REAL_WEIGHT_PATH = os.path.join(MODEL_DIR, 'real_actor_weights.pth')
SIM_CRITIC1_WEIGHT_PATH = os.path.join(MODEL_DIR, 'sim_critic1_weights.pth')
SIM_CRITIC2_WEIGHT_PATH = os.path.join(MODEL_DIR, 'sim_critic2_weights.pth')
REAL_CRITIC1_WEIGHT_PATH = os.path.join(MODEL_DIR, 'real_critic1_weights.pth')
REAL_CRITIC2_WEIGHT_PATH = os.path.join(MODEL_DIR, 'real_critic2_weights.pth')

# 训练模式由 experiment_config.py 中的 USER_STAGE 统一确定。
TRAIN_COUNT_PATH = os.path.join(MODEL_DIR, 'train_count.txt')

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 奖励权重 ====================
# 对应：yr, zr, rh, dx, dy, dt
# rh(过网高度) 权重最高，确保球能过网
REWARD_WEIGHTS = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 0.1]

# ==================== 训练配置 ====================
TRAIN_INTERVAL = 20  # 每 20 轮训练更新一次权重
#BATCH_SIZE = 32
if TRAIN_MODE == 'sim':
    LEARNING_RATE_ACTOR = 1e-3
    LEARNING_RATE_CRITIC = 1e-3
    TRAIN_EPOCHS_PER_UPDATE = 30
    BATCH_SIZE = 128
    DATA_AUGMENTATION = False          # 仿真数据量大，不需要增强
    AUGMENT_NOISE_STD = 0.02           # 占位，实际不会使用
    EPS = 1e-8
else:
    LEARNING_RATE_ACTOR = 1e-3
    LEARNING_RATE_CRITIC = 1e-3
    TRAIN_EPOCHS_PER_UPDATE = 30
    BATCH_SIZE = 8                    # 小 batch 防止过拟合
    DATA_AUGMENTATION = True           # 实际数据增强
    AUGMENT_NOISE_STD = 0.02           # 添加高斯噪声的标准差（相对值）
    EPS = 1e-8
# ==================== 网络配置 ====================
CONFIG = {
    'TRAIN_MODE': TRAIN_MODE,
    'USE_PREDICTED_STATE': False,  # False: 版本 1(仅初始), True: 版本 2(初始 + 预测)
    'STATE_DIM_V1': 9,            #[hit_x, hit_y, hit_z, vx, vy, vz, target_x, target_y, target_v]
    'STATE_DIM_V2': 9,
    'ACTION_DIM': 7,               # [px, py, pz, rx, ry, k/t_delay, vx_cmd]
    'REWARD_DIM': 7,               # [dy, dz, rt, rh, px, py, dv]
    'HIDDEN_DIM': 256,
    'DEVICE': 'cuda',              # 或 'cpu'
    'DATA_AUGMENTATION': DATA_AUGMENTATION,   # 在 if-else 后定义
    'AUGMENT_NOISE_STD': AUGMENT_NOISE_STD,
}
# ==================== 动作空间限制 ====================
# 机械臂工作空间限制（根据实际机械臂调整）
'''
ACTION_LIMITS = {
    'px': (1.5, 1.7),      # 米
    'py': (-0.35, 0.35),     # 米
    'pz': (0.1, 0.6),      # 米
    'rx': (-3.14/4, 3.14/4),  # 弧度
    'ry': (-3.14/4, 3.14/4),
    't_delay': (0.1, 1.0),     # 秒
    'vx_cmd': (0.5, 3.0),      # m/s
}
'''
