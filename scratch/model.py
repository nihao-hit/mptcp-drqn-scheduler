
import random
from collections import namedtuple
from itertools import count
import json

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

State = namedtuple('State', ("s1IsAssoc", "s1WifiRate", "s1Snr", "s1Rtt", "s1UnAckPkts", "s1Retx", 
                             "s2IsAssoc", "s2WifiRate", "s2Snr", "s2Rtt", "s2UnAckPkts", "s2Retx"))
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextState', 'timestamp'))

class ReplayMemory:

    def __init__(self, capacity, filePath):
        # self.capacity = capacity # 暂时没有使用capacity
        self.filePath = filePath
        # self.position = 0 # 暂时没有使用position
        self.memory = []
    
    # ReplayMemory的更新为调用read()重新全部读取transition.json文件
    def read(self):
        self.memory = []
        with open(self.filePath, 'r') as f:
            for line in f.readlines():
                tranDict = json.loads(line)

                # TODO: 这里不是复制有没有问题
                stateDict = tranDict['state']
                nextStateDict = tranDict['nextState']
                for i in range(len(stateDict)):
                    tranDict['state'][i] = State(**stateDict[i])
                    tranDict['nextState'][i] = State(**nextStateDict[i])
                
                self.memory.append(Transition(**tranDict))

    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)

class DRQN(nn.Module):

    def __init__(self, featNums, lstmLayers, subNums):
        super(DRQN, self).__init__()
        # 由于一条Transition元组本身就聚合了时间序列状态，因此lstm输入应以batchSize为第一维度。
        self.lstm = nn.LSTM(input_size=featNums, hidden_size=2 * featNums, num_layers=lstmLayers,
                            batch_first=True)
        self.linear1 = nn.Linear(2 * featNums, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, subNums)
    
    def forward(self, x, h_t, c_t):
        _, (h_t, c_t) = self.lstm(x, (h_t, c_t))
        x = F.relu(self.linear1(h_t[-1]))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # 这里返回的x不是action向量，而是action向量对应的Q值向量
        return x, h_t, c_t


if __name__ == '__main__':
    # ################################################################################
    # # 用于调试libtorch
    # input = torch.rand(128, 8, 12)
    # # 注意：batch_first=true只会调整input, output的形状，h_t, c_t不变
    # h_t = torch.rand(2, 128, 24)
    # c_t = torch.rand(2, 128, 24)
    # scriptModel = torch.jit.trace(DRQN(12, 2, 2), (input, h_t, c_t))
    # scriptModel.save('/home/cx/Desktop/drqn.pt')
    # ################################################################################

    ################################################################################
    # 超参数配置
    filePath = r'/home/cx/Desktop/transition.json'
    memSize = 1e6

    memory = ReplayMemory(memSize, filePath)

    featNums = 12
    lstmSeqLen = 8
    lstmLayers = 2
    subNums = 2
    policyNet = DRQN(featNums=featNums, lstmLayers=lstmLayers, subNums=subNums)
    targetNet = DRQN(featNums=featNums, lstmLayers=lstmLayers, subNums=subNums)
    targetNet.load_state_dict(policyNet.state_dict())
    targetNet.eval() # model.eval()与model.train()对应，前者在评估模型时调用，后者在训练模型时调用，作用是开关某些特殊层

    lr = 0.001
    optimizer = optim.SGD(policyNet.parameters(), lr=lr)
    ################################################################################

    ################################################################################
    # 训练
    episodes = 1000
    batchSize = 128
    gamma = 0.999
    memoryUpdate = 500
    # targetUpdate = 10

    for i in range(episodes):
        for j in count():
            if j % memoryUpdate == 0: # 回合中每训练memoryUpdate次更新一次ReplayMemory
                memory.read()
            if len(memory) < batchSize:
                continue
            transitions = memory.sample(batchSize)
            # 转置批样本(有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。
            # 这会将转换的批处理数组转换为批处理数组的转换。
            batch = Transition(*zip(*transitions))

            stateB = torch.tensor(batch.state).to(dtype=torch.float32)
            actionB = torch.tensor(batch.action)
            rewardB = torch.tensor(batch.reward)
            # TODO: 目前没有确定终态，可以不设置终态，而通过判断loss低于阈值作为回合结束吗？
            nextStateB = torch.tensor(batch.nextState).to(dtype=torch.float32)
            
            # TODO: h_t, c_t初始化值确定
            h_t = Variable(torch.zeros(lstmLayers, batchSize, 2 * featNums))
            c_t = Variable(torch.zeros(lstmLayers, batchSize, 2 * featNums))
            # 计算Q(s_t, a)-模型计算 Q(s_t)，然后选择所采取行动的列。这些是根据策略网络对每个批处理状态所采取的操作。
            # TODO: policy网络返回的h_t, c_t需要传给target网络吗？
            policyAction, _, _ = policyNet(stateB, h_t, c_t)
            policyQvalue = policyAction.gather(1, actionB.unsqueeze(1))

            # 计算下一个状态的V(s_{t+1})。非最终状态下一个状态的预期操作值是基于“旧”目标网络计算的；
            # 选择max(1)[0]的最佳奖励。这是基于掩码合并的，这样当状态为最终状态时，我们将获得预期状态值或0。
            nextStateQvalue, _, _ = targetNet(nextStateB, h_t, c_t)
            nextStateQvalue = nextStateQvalue.max(1)[0].detach()
            # 计算期望 Q 值
            expectedQvalue = (nextStateQvalue * gamma) + rewardB

            # # 计算 Huber 损失
            # loss = F.smooth_l1_loss(policyQvalue, expectedQvalue.unsqueeze(1))
            # 计算均方误差
            loss = F.mse_loss(policyQvalue, expectedQvalue.unsqueeze(1))
            
            # 优化模型
            optimizer.zero_grad()
            loss.backward()
            # TODO: clamp()避免参数出现inf，这步操作是不是必要的？
            for param in policyNet.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            if j % 100 == 0:
                print('j={}, loss={:.5f}'.format(j, loss.item()))
            # 判断loss低于阈值作为终态
            if j == 1000:
                break
        #每回合结束后更新目标网络, 复制在DQN中的所有权重偏差
        targetNet.load_state_dict(policyNet.state_dict())

        # 序列化targetNet
        input = torch.rand(batchSize, lstmSeqLen, featNums)
        # 注意：batch_first=true只会调整input, output的形状，h_t, c_t不变
        h_t = torch.rand(lstmLayers, batchSize, 2*featNums)
        c_t = torch.rand(lstmLayers, batchSize, 2*featNums)
        scriptModel = torch.jit.trace(targetNet, (input, h_t, c_t))
        scriptModel.save('/home/cx/Desktop/drqn.pt')
    ################################################################################