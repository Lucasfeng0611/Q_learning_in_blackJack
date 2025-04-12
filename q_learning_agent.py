# q_learning_agent.py
import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.01,
                 epsilon_decay=0.99995, min_epsilon=0.01,
                 alpha_decay=0.9999, min_alpha=0.01):
        self.q_table = defaultdict(lambda: [0, 0])
        self.alpha = alpha          # 初始学习率更高
        self.gamma = gamma          # 更关注短期收益
        self.epsilon = epsilon      # 初始探索率设为100%
        self.epsilon_decay = epsilon_decay  # 衰减系数
        self.min_epsilon = min_epsilon      # 最小探索率
        self.alpha_decay = alpha_decay      # 学习率衰减
        self.min_alpha = min_alpha          # 最小学习率

    def get_state_representation(self, raw_state):
        """直接返回增强后的状态元组"""
        return raw_state  # 原样返回所有维度信息

    def get_action(self, state):
        """使用ε-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])  # 随机选择
        else:
            return np.argmax(self.q_table[state])  # 选择最优动作

    def update(self, state, action, reward, next_state, done):
        """更新Q值"""
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (target - current_q)