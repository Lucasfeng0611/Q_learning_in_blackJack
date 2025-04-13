# q_learning_agent.py
import numpy as np
import pickle
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.01,
                 epsilon_decay=0.99995, min_epsilon=0.01,
                 alpha_decay=0.9999, min_alpha=0.01):
        # 初始化 Q 表，状态 -> [Q(停牌), Q(要牌)]
        self.q_table = defaultdict(lambda: [0.0, 0.0])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha

    def get_state_representation(self, raw_state):
        """当前直接返回原始状态元组，可在此处进一步处理/离散化"""
        return raw_state

    def imitation_learning(self, strategy_data, num_epochs=10, expert_bonus=1.0):
        for _ in range(num_epochs):
            for state, expert_action in strategy_data:
                # 如果状态表示 player_total 小于12，我们对专家动作给予更高奖励
                player_total = state[0]
                bonus = expert_bonus * 1.5 if player_total < 12 else expert_bonus
                self.q_table[state] = [
                    -bonus if a != expert_action else bonus
                    for a in [0, 1]
                ]
        print("行为克隆预训练完成.")

    def get_action(self, state):
        """使用ε-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """更新 Q 值"""
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - current_q)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q表已保存到 {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            loaded_q = pickle.load(f)
            self.q_table.update(loaded_q)
        print(f"Q表从 {path} 加载完成")
