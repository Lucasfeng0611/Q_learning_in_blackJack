# train.py
from blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
from basic_strategy import BasicStrategy
import pickle


def generate_strategy_data(num_random=10000):
    """
    生成专家示范数据。既包含通过环境 reset 获取的随机状态，
    也通过遍历部分状态空间收集专家决策数据。
    """
    env = BlackjackEnv()
    strategy_data = []

    # 随机采样部分
    for _ in range(num_random):
        state = env.reset()
        expert_action = BasicStrategy.get_action(state)
        strategy_data.append((state, expert_action))

    # 遍历部分状态空间（玩家总点数 4～21，庄家明牌 2～11；usable_ace:0/1）
    for player_pts in range(4, 22):
        for dealer_pts in range(2, 12):
            for usable_ace in [0, 1]:
                state = (player_pts, dealer_pts, usable_ace)
                expert_action = BasicStrategy.get_action(state)
                strategy_data.append((state, expert_action))

    print(f"生成专家数据总量: {len(strategy_data)}")
    return strategy_data


def train(num_episodes=10000000):
    # 第一阶段：行为克隆预训练
    agent = QLearningAgent(alpha=0.5, gamma=0.95, epsilon=0.01)
    strategy_data = generate_strategy_data(num_random=10000)
    agent.imitation_learning(strategy_data, num_epochs=20, expert_bonus=1.0)
    agent.save('pretrained.pkl')

    # 第二阶段：强化学习微调
    env = BlackjackEnv()
    agent.load('pretrained.pkl')
    # 恢复为正常 RL 参数
    agent.alpha = 0.2
    agent.epsilon = 0.1

    wins = 0
    losses = 0
    draws = 0

    for episode in range(num_episodes):
        raw_state = env.reset()
        state = agent.get_state_representation(raw_state)
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            next_raw_state, reward, done = env.step(action)
            next_state = agent.get_state_representation(next_raw_state)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                episode_reward = reward

        # 根据奖励区分胜负（这里假设超过0.25为胜利，小于0为失败，其余为平局）
        if episode_reward > 0.25:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1

        if (episode + 1) % 10000 == 0:
            total = wins + losses + draws
            loss_rate = (losses / total) * 100 if total > 0 else 0
            print(f"Episode {episode + 1} | 输率: {loss_rate:.2f}% (胜: {wins} 负: {losses} 平: {draws})")
            wins = losses = draws = 0
            print(f"Episode {episode + 1} completed")

        # 衰减探索率与学习率
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        agent.alpha = max(agent.min_alpha, agent.alpha * agent.alpha_decay)

    agent.save('trained_agent.pkl')
    return agent


if __name__ == "__main__":
    trained_agent = train()
