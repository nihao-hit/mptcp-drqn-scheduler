import math
import pandas as pd
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'timestamp'))
# 由于policyNet与targetNet异步交互，因此以csv文件链接
class ReplayMemory:

    def __init__(self, capacity, filePath):
        self.capacity = capacity
        self.filePath = filePath
        self.position = 0
        self.memory = []
        self.read()
    
    def read(self):
        self.memory = list(map(lambda tuple: Transition(*tuple), 
                               pd.DataFrame(self.filePath).values.tolist()))
        self.position = len(self.memory) - 1
    
    def write(self):
        pd.DataFrame(self.memory, columns=['state', 'action', 'next_state', 'reward', 'timestamp']).to_csv(self.filePath)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DRQN(nn.Module):

    def __init__(self, featNum, numLayers, subNum):
        super(DRQN, self).__init__()
        # 由于一条Transition元组本身就聚合了时间序列状态，因此lstm输入应以batch_size为第一维度。
        self.lstm = nn.LSTM(input_size=featNum, hidden_size=2 * featNum, num_layers=numLayers,
                            batch_first=True)
        self.linear1 = nn.Linear(2 * featNum, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, subNum)
    
    def forward(self, x, h_t, h_c):
        # 这里x形状还需要调整以适应lstm
        _, (h_t, c_t) = self.lstm(x, (h_t, h_c))
        x = F.relu(self.linear1(h_t))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return F.softmax(x), h_t, c_t
# features(子流平均吞吐量，子流平均拥塞窗口、子流平均时延、子流累计未确认包数量、子流重传包数量)
featNum = 10
# 暂时固定网络状态时间序列长度
seqNum = 8
subNum = 2
numLayers = 2
policy_net = DRQN(featNum=featNum, numLayers=numLayers, subNum=subNum)
target_net = DRQN(featNum=featNum, numLayers=numLayers, subNum=subNum)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# targetNet一次在线采集1000条经验元组，policyNet一回合使用1000条经验元组训练，
# 而经验库可以存放1000回合的经验元组。
memSize = 1e6
episode = 1e3
filePath = r'X:\Desktop\my-drqn\data.csv'
memory = ReplayMemory(memSize, filePath)

# 训练，训练后保存模型
lr = 0.001
# lossFn = nn.MSELoss()
optimizer = optim.SGD(policy_net.parameters(), lr=lr)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1）将为每行的列返回最大值。max result的第二列是找到max元素的索引，因此我们选择预期回报较大的操作。
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 转置批样本(有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。这会将转换的批处理数组转换为批处理数组的转换。
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接批处理元素(最终状态将是模拟结束后的状态）
    # 到达最终状态后，Transition.next_state为None
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 每次批训练都要初始化hidden吗？
    h_t = Variable(torch.zeros(numLayers, BATCH_SIZE, 2 * featNum))
    h_c = Variable(torch.zeros(numLayers, BATCH_SIZE, 2 * featNum))
    # 计算Q(s_t, a)-模型计算 Q(s_t)，然后选择所采取行动的列。这些是根据策略网络对每个批处理状态所采取的操作。
    policyAction, h_t, h_c = policy_net(state_batch, h_t, h_c)
    state_action_values = policyAction.gather(1, action_batch)

    # 计算下一个状态的V(s_{t+1})。非最终状态下一个状态的预期操作值是基于“旧”目标网络计算的；
    # 选择max(1)[0]的最佳奖励。这是基于掩码合并的，这样当状态为最终状态时，我们将获得预期状态值或0。
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算期望 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # # 计算 Huber 损失
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # 计算均方误差
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    # 这步操作是不是必要的？
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # 更新目标网络
    

num_episodes = 1000
for i_episode in range(num_episodes):
    for t in count():
        # 选择并执行动作
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 观察新状态
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 在内存中储存当前参数
        memory.push(state, action, next_state, reward)

        # 进入下一状态
        state = next_state

        # 记性一步优化 (在目标网络)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    #更新目标网络, 复制在 DQN 中的所有权重偏差
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())