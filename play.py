# play.py
import pygame
from blackjack_env import BlackjackEnv
import pickle
from collections import defaultdict
from q_learning_agent import QLearningAgent

class BlackjackGame:
    def __init__(self, agent=None):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("21点游戏")
        self.font = pygame.font.Font(None, 30)
        self.env = BlackjackEnv()
        self.agent = agent
        self.colors = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'GREEN': (0, 128, 0),
            'GRAY': (200, 200, 200)
        }
        # 添加重启按钮属性
        self.restart_btn = pygame.Rect(600, 500, 150, 50)
        self.result_text = None  # 新增结果文本属性

    def draw_hand(self, hand, x, y):
        for i, card in enumerate(hand):
            text = self.font.render(f"{card[0]}{card[1]}", True, self.colors['WHITE'])
            self.screen.blit(text, (x + i * 100, y))

    def run(self):
        state = self.env.reset()
        done = False
        clock = pygame.time.Clock()

        while True:
            self.screen.fill(self.colors['GREEN'])

            # 绘制重启按钮
            pygame.draw.rect(self.screen, self.colors['GRAY'], self.restart_btn)
            btn_text = self.font.render("RESTART", True, self.colors['BLACK'])
            self.screen.blit(btn_text, (610, 515))

            # 绘制手牌和点数
            self.draw_hand(self.env.player_hand, 100, 400)
            self.draw_hand(self.env.dealer_hand, 100, 100)

            # 显示点数
            player_text = self.font.render(
                f"Player: {self.env.calculate_hand_value(self.env.player_hand)[0]}",
                True, self.colors['WHITE'])
            dealer_text = self.font.render(
                f"Dealer: {self.env.calculate_hand_value(self.env.dealer_hand)[0]}",
                True, self.colors['WHITE'])
            self.screen.blit(player_text, (100, 350))
            self.screen.blit(dealer_text, (100, 50))

            # AI自动决策
            if self.agent and not done:
                action = self.agent.get_action(state)
                state, reward, done = self.env.step(action)
                if done:
                    # 根据结果生成文本
                    if reward > 0.25:
                        self.result_text = self.font.render("Player Win!", True, self.colors['WHITE'])
                    elif reward < 0:
                        self.result_text = self.font.render("Dealer win!", True, self.colors['WHITE'])
                    else:
                        self.result_text = self.font.render("Draw!", True, self.colors['WHITE'])

            # 绘制游戏结果
            if self.result_text:
                text_rect = self.result_text.get_rect(center=(400, 300))
                self.screen.blit(self.result_text, text_rect)

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.restart_btn.collidepoint(event.pos):
                        state = self.env.reset()
                        done = False
                        self.result_text = None  # 重置时清除结果

            pygame.display.update()
            clock.tick(30)

if __name__ == "__main__":
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)
        agent = QLearningAgent()
        agent.q_table = defaultdict(lambda: [0,0], q_table)

    game = BlackjackGame(agent)
    game.run()