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
        self.restart_btn = pygame.Rect(600, 500, 150, 50)
        self.result_text = None

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
            pygame.draw.rect(self.screen, self.colors['GRAY'], self.restart_btn)
            btn_text = self.font.render("RESTART", True, self.colors['BLACK'])
            self.screen.blit(btn_text, (610, 515))

            self.draw_hand(self.env.player_hand, 100, 400)
            self.draw_hand(self.env.dealer_hand, 100, 100)

            player_text = self.font.render(
                f"Player: {self.env.calculate_hand_value(self.env.player_hand)[0]}",
                True, self.colors['WHITE'])
            dealer_text = self.font.render(
                f"Dealer: {self.env.calculate_hand_value(self.env.dealer_hand)[0]}",
                True, self.colors['WHITE'])
            self.screen.blit(player_text, (100, 350))
            self.screen.blit(dealer_text, (100, 50))

            if self.agent and not done:
                action = self.agent.get_action(state)
                state, reward, done = self.env.step(action)
                if done:
                    if reward > 0.25:
                        self.result_text = self.font.render("Player Win!", True, self.colors['WHITE'])
                    elif reward < 0:
                        self.result_text = self.font.render("Dealer win!", True, self.colors['WHITE'])
                    else:
                        self.result_text = self.font.render("Draw!", True, self.colors['WHITE'])

            if self.result_text:
                text_rect = self.result_text.get_rect(center=(400, 300))
                self.screen.blit(self.result_text, text_rect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.restart_btn.collidepoint(event.pos):
                        state = self.env.reset()
                        done = False
                        self.result_text = None

            pygame.display.update()
            clock.tick(1000)

if __name__ == "__main__":
    with open('trained_agent.pkl', 'rb') as f:
        q_table = pickle.load(f)
        agent = QLearningAgent()
        agent.q_table = defaultdict(lambda: [0.0, 0.0], q_table)

    # 批量测试模式
    test_mode = True  # 设为False可恢复GUI游玩模式

    if test_mode:
        env = BlackjackEnv()
        win, lose, draw = 0, 0, 0
        for _ in range(10000):
            state = env.reset()
            done = False
            while not done:
                action = agent.get_action(state)
                state, reward, done = env.step(action)
            if reward > 0.25:
                win += 1
            elif reward < 0:
                lose += 1
            else:
                draw += 1
        print(f"胜局: {win}, 负局: {lose}, 平局: {draw}")
        print(f"胜率: {win / 100:.2f}%, 输率: {lose / 100:.2f}%, 平局率: {draw / 100:.2f}%")
    else:
        # 原有GUI模式
        game = BlackjackGame(agent)
        game.run()