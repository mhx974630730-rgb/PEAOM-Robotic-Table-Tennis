# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tanhnormal import TanhNormal
from config import CONFIG

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        s_dim = CONFIG['STATE_DIM_V2'] if CONFIG['USE_PREDICTED_STATE'] else CONFIG['STATE_DIM_V1']
        self.shared = nn.Sequential(
            nn.Linear(s_dim + CONFIG['ACTION_DIM'], CONFIG['HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(CONFIG['HIDDEN_DIM'], CONFIG['HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(CONFIG['HIDDEN_DIM'], CONFIG['HIDDEN_DIM']),
            nn.ReLU()
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-1e-3, 1e-3)
                m.bias.data.fill_(0)
        self.shared.apply(init_weights)
        self.out = nn.Linear(256, CONFIG['REWARD_DIM'])
        self.out.weight.data.uniform_(-1e-3,1e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        feat = self.shared(x)
        actions_value = self.out(feat)
        return actions_value

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        s_dim = CONFIG['STATE_DIM_V2'] if CONFIG['USE_PREDICTED_STATE'] else CONFIG['STATE_DIM_V1']
        self.net = nn.Sequential(
            nn.Linear(s_dim, CONFIG['HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(CONFIG['HIDDEN_DIM'], CONFIG['HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(CONFIG['HIDDEN_DIM'], CONFIG['HIDDEN_DIM']),
            nn.ReLU()
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-1e-3, 1e-3)
                m.bias.data.fill_(0)
        self.net.apply(init_weights)
        self.out_mean = nn.Linear(CONFIG['HIDDEN_DIM'], CONFIG['ACTION_DIM'])
        self.out_mean.weight.data.uniform_(-1e-3,1e-3) # initilizaiton of OUT
        self.out_std = nn.Linear(CONFIG['HIDDEN_DIM'], CONFIG['ACTION_DIM'])
        self.out_std.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state,reparameterize=True, deterministic=False, return_log_prob=False):
        x = self.net(state)
        mean = self.out_mean(x)
        log_std = self.out_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                else:
                    action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
        return action ,log_prob

def weighted_l2_loss(predicted_reward):
    return torch.sum(predicted_reward ** 2, dim=1, keepdim=True)
