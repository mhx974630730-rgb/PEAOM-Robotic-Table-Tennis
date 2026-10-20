#!/usr/bin/env python3
# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import glob
import time
import argparse
import fcntl
import json
import rospy   # 添加 rospy 支持
import sys
from experiment_config import (
    IS_REAL_ROBOT, ALLOW_REAL_FINETUNE, RANDOM_SEED, RUN_DIR,
    ensure_run_dir, write_metadata,
)
from config import (
    CONFIG, SIM_WEIGHT_PATH, SIM_CRITIC1_WEIGHT_PATH, SIM_CRITIC2_WEIGHT_PATH, REAL_WEIGHT_PATH, REAL_CRITIC1_WEIGHT_PATH, REAL_CRITIC2_WEIGHT_PATH,
    DATA_DIR, TRAIN_COUNT_PATH, TRAIN_INTERVAL, BATCH_SIZE,
    LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC, TRAIN_EPOCHS_PER_UPDATE,
    EPS
)
from network import Actor, Critic, weighted_l2_loss
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward):
        """添加单个样本"""
        self.buffer.append((state, action, reward))

    def extend(self, states, actions, rewards):
        """批量添加样本（states, actions, rewards 为 numpy 数组，每行一个样本）"""
        for i in range(len(states)):
            self.buffer.append((states[i], actions[i], rewards[i]))

    def sample(self, batch_size):
        """随机采样一批样本"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards)

    def __len__(self):
        return len(self.buffer)
class Trainer(nn.Module):
    def __init__(self,
                 new_training=True,
                 use_automatic_entropy_tuning=False,
                 alpha=0.01,
                 max_train_count=10000):
        super(Trainer, self).__init__()
        # Seed every random source used by network initialization, action
        # sampling, and replay-buffer mini-batch sampling.
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
        # 根据训练模式选择权重路径
        if CONFIG.get('TRAIN_MODE', 'sim') == 'sim':
            self.actor_path = SIM_WEIGHT_PATH
            self.critic1_path = SIM_CRITIC1_WEIGHT_PATH
            self.critic2_path = SIM_CRITIC2_WEIGHT_PATH
            self.init_from_real = False
        else:
            self.actor_path = REAL_WEIGHT_PATH
            self.critic1_path = REAL_CRITIC1_WEIGHT_PATH
            self.critic2_path = REAL_CRITIC2_WEIGHT_PATH
            self.init_from_real = True

        self.device = torch.device(CONFIG['DEVICE'] if torch.cuda.is_available() else 'cpu')
        self.actor = Actor().to(self.device)
        self.critic1 = Critic().to(self.device)
        self.critic2 = Critic().to(self.device)
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR,eps = EPS)
        self.opt_critic1 = optim.Adam(self.critic1.parameters(), lr=LEARNING_RATE_CRITIC,eps = EPS)
        self.opt_critic2 = optim.Adam(self.critic2.parameters(), lr=LEARNING_RATE_CRITIC,eps = EPS)
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.action_size = CONFIG['ACTION_DIM']
        self.alpha = torch.tensor(alpha, device=self.device)
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -self.action_size
            self.alpha_optim = optim.Adam([self.log_alpha], lr=LEARNING_RATE_ACTOR)

        #self.weights = torch.tensor(REWARD_WEIGHTS, dtype=torch.float32).to(self.device)

        # 获取当前训练计数（如果是新训练，则重置计数）
        self.train_count = self.load_train_count(reset=new_training)

        # 加载已有权重（仅当不是新训练时）
        if not new_training:
            self.load_existing_weights()
        else:
            rospy.loginfo("Starting new training from scratch (no weights loaded).")
            if os.path.exists(self.actor_path):
                os.remove(self.actor_path)
                rospy.loginfo(f"Removed existing actor weights: {self.actor_path}")
            if os.path.exists(self.critic1_path):
                os.remove(self.critic1_path)
                rospy.loginfo(f"Removed existing critic weights: {self.critic1_path}")
            if os.path.exists(self.critic2_path):
                os.remove(self.critic2_path)
                rospy.loginfo(f"Removed existing critic weights: {self.critic2_path}")
            self.save_train_count()

        rospy.loginfo(f"Trainer initialized on {self.device}")
        rospy.loginfo(f"Current train count: {self.train_count}")
        self.critic_loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=5000)   # 缓冲区容量，可根据内存调整
        self.samples_since_last_train = 0
        self.train_interval_samples = 1   # 每累积 2 个新样本训练一次
        script_dir = os.path.dirname(os.path.abspath(__file__))
        loss_dir = os.path.join(script_dir, "loss_data")
        os.makedirs(loss_dir, exist_ok=True)
        self.loss_file = os.path.join(loss_dir, "loss.txt")
        with open(self.loss_file, "w") as f:
            f.write("")
        #debug
        self.phase2_start_count = 200
        self.phase2 = False
        self.max_train_count = max_train_count
        self.write_training_metadata()

    def write_training_metadata(self):
        """Persist the effective optimizer/training settings for this run."""
        ensure_run_dir()
        payload = {
            'random_seed': RANDOM_SEED,
            'device': str(self.device),
            'actor_learning_rate': LEARNING_RATE_ACTOR,
            'critic_learning_rate': LEARNING_RATE_CRITIC,
            'batch_size': BATCH_SIZE,
            'epochs_per_update': TRAIN_EPOCHS_PER_UPDATE,
            'replay_buffer_capacity': self.replay_buffer.capacity,
            'new_samples_per_update': self.train_interval_samples,
            'weight_save_interval_updates': TRAIN_INTERVAL,
            'maximum_training_updates': self.max_train_count,
            'phase2_start_update': self.phase2_start_count,
            'automatic_entropy_tuning': self.use_automatic_entropy_tuning,
            'initial_alpha': float(self.alpha.detach().cpu().item()),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
        }
        path = os.path.join(str(RUN_DIR), 'training_metadata.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


    def load_existing_weights(self):
        if os.path.exists(self.actor_path):
            self.actor.load_state_dict(torch.load(self.actor_path))
            rospy.loginfo(f"Loaded actor weights from {self.actor_path}")
        elif self.init_from_real and os.path.exists(SIM_WEIGHT_PATH):
            self.actor.load_state_dict(torch.load(SIM_WEIGHT_PATH))
            rospy.loginfo("Initialized real actor from sim weights")

        if os.path.exists(self.critic1_path):
            self.critic1.load_state_dict(torch.load(self.critic1_path))
            rospy.loginfo(f"Loaded critic1 weights from {self.critic1_path}")
        elif self.init_from_real and os.path.exists(SIM_CRITIC1_WEIGHT_PATH):
            self.critic1.load_state_dict(torch.load(SIM_CRITIC1_WEIGHT_PATH))
            rospy.loginfo("Initialized real critic1 from sim weights")

        if os.path.exists(self.critic2_path):
            self.critic2.load_state_dict(torch.load(self.critic2_path))
            rospy.loginfo(f"Loaded critic2 weights from {self.critic2_path}")
        elif self.init_from_real and os.path.exists(SIM_CRITIC2_WEIGHT_PATH):
            self.critic2.load_state_dict(torch.load(SIM_CRITIC2_WEIGHT_PATH))
            rospy.loginfo("Initialized real critic2 from sim weights")

    def load_train_count(self, reset=True):
        if reset:
            return 0
        if os.path.exists(TRAIN_COUNT_PATH):
            with open(TRAIN_COUNT_PATH, 'r') as f:
                return int(f.read().strip())
        return 0

    def save_train_count(self):
        with open(TRAIN_COUNT_PATH, 'w') as f:
            f.write(str(self.train_count))

    def load_data(self):
        """加载新数据并加入缓冲区，返回新增的样本数"""
        files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
        total_new_samples = 0
        if not files:
            return 0
        #processed_dir = os.path.join(DATA_DIR, "processed")
        #os.makedirs(processed_dir, exist_ok=True)

        for f in files:
            try:
                with open(f, 'rb') as fd:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    data = np.load(fd, allow_pickle=True)
                    if 'states' not in data or 'actions' not in data or 'rewards' not in data:
                        rospy.logwarn(f"Skipping {f}: missing fields")
                        continue
                    s = np.array(data['states'], dtype=np.float32).reshape(-1, 9)
                    #rospy.loginfo(f"Loaded states shape: {s.shape}")  # 添加这行
                    a = np.array(data['actions'], dtype=np.float32).reshape(-1, 7)
                    r = np.array(data['rewards'], dtype=np.float32).reshape(-1, 7)
                    #rospy.loginfo(f'state:{s}')
                    #rospy.loginfo(f'action:{a}')
                    #rospy.loginfo(f'reward:{r}')
                    # 加入缓冲区
                    self.replay_buffer.extend(s, a, r)
                    total_new_samples += len(s)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    # 移动文件
                os.remove(f)
            except Exception as e:
                rospy.logerr(f"Error loading {f}: {e}")
                error_dir = os.path.join(DATA_DIR, "error")
                os.makedirs(error_dir, exist_ok=True)
                os.rename(f, os.path.join(error_dir, os.path.basename(f)))
                continue
        rospy.loginfo(f"Found {len(files)} npz files, loaded {total_new_samples} samples")
        return total_new_samples

    def train_step(self, states, actions, rewards):
        self.train()
        rewards = torch.clamp(rewards, -10, 10)
        """ 单步训练 """
        # 1. 更新 Critic: 预测奖励逼近真实奖励


        if not self.phase2:
            q1_pred = self.critic1(states, actions)[:, :2]
            q2_pred = self.critic2(states, actions)[:, :2]
            target = rewards[:, :2]
        else:
            q1_pred = self.critic1(states, actions)
            q2_pred = self.critic2(states, actions)
            target = rewards

        value_loss1 = self.critic_loss(target, q1_pred)
        #value_loss1 = value_loss1.mean()
        value_loss2 = self.critic_loss(target, q2_pred)
        #value_loss2 = value_loss2.mean()

        self.opt_critic1.zero_grad()
        value_loss1.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.opt_critic1.step()

        self.opt_critic2.zero_grad()
        value_loss2.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.opt_critic2.step()

        # 2. 更新 Actor: 最小化 Critic 预测的加权奖励范数
        self.opt_actor.zero_grad()
        new_actions, log_prob = self.actor(states, reparameterize=True, return_log_prob=True)
        q1_n = self.critic1(states, new_actions)
        q2_n = self.critic2(states, new_actions)
        if not self.phase2:
            q1_n = q1_n[:, :2]
            q2_n = q2_n[:, :2]

        w1 = weighted_l2_loss(q1_n)   # [batch,1]
        w2 = weighted_l2_loss(q2_n)   # [batch,1]
        min_q = torch.min(w1, w2)
        policy_loss = (self.alpha * log_prob + min_q).mean()


        policy_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_actor.step()


        #更新 Alpha
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = 0

        action_loss = policy_loss.item()
        v1_loss = value_loss1.item()
        v2_loss = value_loss2.item()
        with open(self.loss_file, "a") as f:
            f.write(f"{(v1_loss + v2_loss)/2},{action_loss}\n")
        return (v1_loss + v2_loss)/2, action_loss

    def save_weights_safely(self):
        torch.save(self.actor.state_dict(), self.actor_path + ".tmp")
        torch.save(self.critic1.state_dict(), self.critic1_path + ".tmp")
        torch.save(self.critic2.state_dict(), self.critic2_path + ".tmp")
        os.replace(self.actor_path + ".tmp", self.actor_path)
        os.replace(self.critic1_path + ".tmp", self.critic1_path)
        os.replace(self.critic2_path + ".tmp", self.critic2_path)
        rospy.loginfo(f"Weights saved (train_count={self.train_count})")

    def run(self):
        rospy.loginfo("Starting training loop with replay buffer...")
        rospy.loginfo(f"Buffer capacity: {self.replay_buffer.capacity}")
        rospy.loginfo(f"Train every {self.train_interval_samples} new samples")

        while not rospy.is_shutdown():
            # 加载新数据并加入缓冲区
            new_samples = self.load_data()
            if new_samples > 0:
                self.samples_since_last_train += new_samples
                #rospy.loginfo(f"Added {new_samples} samples. Buffer size: {len(self.replay_buffer)}")
                #sys.stdout.write(f"Added {new_samples} samples. Buffer size: {len(self.replay_buffer)}")
            # 如果缓冲区样本数足够，且达到训练间隔，则进行训练
            #rospy.loginfo(f"Buffer size: {len(self.replay_buffer)}, samples_since_last_train: {self.samples_since_last_train}")
            if len(self.replay_buffer) >= BATCH_SIZE and self.samples_since_last_train >= self.train_interval_samples:
                rospy.loginfo("Training condition met, entering training loop")
                # 训练多个 epoch（例如 1 个 epoch 或固定步数）
                for epoch in range(TRAIN_EPOCHS_PER_UPDATE):
                    states, actions, rewards = self.replay_buffer.sample(BATCH_SIZE)
                    # 转换为 tensor
                    states_t = torch.FloatTensor(states).to(self.device)
                    actions_t = torch.FloatTensor(actions).to(self.device)
                    rewards_t = torch.FloatTensor(rewards).to(self.device)
                    lc, la = self.train_step(states_t, actions_t, rewards_t)
                    rospy.loginfo(f"Train count {self.train_count}: Critic Loss {lc:.4f}, Actor Loss {la:.4f}")
                    #if self.train_count % 20 == 0:
                    #    rospy.loginfo(f"self.train_count {self.train_count}: Critic Loss {lc:.4f}, Actor Loss {la:.4f}")
                # 重置计数器
                self.samples_since_last_train = 0
                # 更新训练计数并定期保存模型
                self.train_count += 1
                self.save_train_count()
                if self.train_count % TRAIN_INTERVAL == 0:
                    rospy.loginfo(f"Saving weights at train_count={self.train_count}")
                    self.save_weights_safely()
                    rospy.loginfo("save_weights_safely")
            if self.train_count >= self.phase2_start_count and not self.phase2:
                self.phase2 = True
                rospy.loginfo("Phase 2 started: now training with full 7_dim reward.")
            # 短暂休眠，避免空转
            if self.train_count >= self.max_train_count:
                rospy.loginfo(f"Reached max train count {self.max_train_count}, stopping training.")
                self.save_weights_safely()
                break   # 退出 while not rospy.is_shutdown()
            time.sleep(0.5)
        rospy.signal_shutdown('Training complete')

if __name__ == '__main__':
    if IS_REAL_ROBOT and not ALLOW_REAL_FINETUNE:
        raise SystemExit(
            "Refusing to train because USER_STAGE='real_eval' and the evaluation is zero-shot. "
            "Do not launch train.py during physical evaluation."
        )
    write_metadata()
    # 初始化 ROS 节点（如果尚未初始化）
    if not rospy.core.is_initialized():
        rospy.init_node('trainer', anonymous=True)
    parser = argparse.ArgumentParser(description='Train RL agent for ping-pong')
    parser.add_argument('--new', action='store_true', help='Start new training')
    args = parser.parse_args()

    new_training = args.new
    try:
        trainer = Trainer(new_training=new_training)
        trainer.run()
    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted")
