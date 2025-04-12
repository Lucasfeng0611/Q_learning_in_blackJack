# train.py
from blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
import pickle


def train(num_episodes=8000000):
    env = BlackjackEnv()
    agent = QLearningAgent()

    # 新增胜率统计变量
    wins = 0
    losses = 0
    draws = 0

    for episode in range(num_episodes):
        raw_state = env.reset()
        state = agent.get_state_representation(raw_state)
        done = False
        episode_reward = 0  # 用于记录最终结果

        while not done:
            action = agent.get_action(state)
            next_raw_state, reward, done = env.step(action)
            next_state = agent.get_state_representation(next_raw_state)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                episode_reward = reward  # 保存最终结果

        # 统计结果
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0 :
            losses += 1
        else:
            draws += 1

        if (episode + 1) % 10000 == 0:
            total = wins + losses + draws
            loss_rate = (losses / total) * 100 if total > 0 else 0
            print(f"Episode {episode + 1} | 输率: {loss_rate:.2f}% (胜: {wins} 负: {losses} 平: {draws})")
            # 重置统计（可选）
            wins = losses = draws = 0
            print(f"Episode {episode + 1} completed")

    agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
    agent.alpha = max(agent.min_alpha, agent.alpha * agent.alpha_decay)
    return agent


if __name__ == "__main__":
    trained_agent = train()
    # 可以保存Q表
    with open('q_table.pkl', 'wb') as f:
         pickle.dump(dict(trained_agent.q_table), f)