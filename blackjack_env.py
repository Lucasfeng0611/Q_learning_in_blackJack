# blackjack_env.py
import random


class BlackjackEnv:
    def __init__(self):
        self.card_values = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
            "J": 10, "Q": 10, "K": 10, "A": 11
        }
        self.reset()

    def get_state(self):
        """返回当前状态（玩家点数, 庄家明牌点数）"""
        return (self.player_value, self.dealer_hand[0][0] if self.dealer_hand else 0)

    def reset(self):
        self.deck = self.create_deck()
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        # 获取玩家初始点数时解包元组
        self.player_value, self.usable_ace = self.calculate_hand_value(self.player_hand)
        return self.get_state()

    def create_deck(self):
        suits = ["♠", "♣", "♦", "♥"]
        ranks = list(self.card_values.keys())
        deck = [(rank, suit) for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand):
        value = 0
        ace_count = 0

        # 计算初始值（A=11）
        for card in hand:
            value += self.card_values[card[0]]
            if card[0] == "A":
                ace_count += 1

        # 调整A的计值（1点）
        while value > 21 and ace_count > 0:
            value -= 10  # 每次将一张A从11变为1（减少10点）
            ace_count -= 1

        # 判断是否有仍可用的A（调整后仍有A为11的情况）
        usable_ace = any(card[0] == "A" for card in hand) and (value <= 21)
        return value, usable_ace

    def get_state(self):
        """返回状态（玩家总点数，庄家明牌点数，是否有可用A）"""
        player_value, usable_ace = self.calculate_hand_value(self.player_hand)
        dealer_showing = self.card_values[self.dealer_hand[0][0]]
        return (player_value, dealer_showing, int(usable_ace))

    def step(self, action):
        done = False
        reward = 0
        base_reward = 0  # 新增基础奖励变量

        # 玩家行动阶段
        if action == 1:  # 要牌
            self.player_hand.append(self.deck.pop())
            player_value, _ = self.calculate_hand_value(self.player_hand)

            # 玩家爆牌立即终止
            if player_value > 21:
                return self.get_state(), -1, True

            return self.get_state(), 0, False

        elif action == 0:  # 停牌
            done = True

            # 获取当前玩家点数（在庄家行动前）
            player_value, _ = self.calculate_hand_value(self.player_hand)

            # 奖励塑形逻辑 -------------------------------------------------
            if player_value >= 17:  # 高风险区域奖励
                # 线性递增：17点+0.05，每增加1点奖励增加0.05，21点达到0.25
                risk_bonus = 0.05 * (player_value - 16)
                base_reward += min(risk_bonus, 0.25)  # 确保不超过0.25

            elif 12 <= player_value <= 16:  # 危险区域惩罚
                # 线性递减惩罚：12点-0.02，每减少1点惩罚减轻，16点不惩罚
                penalty = -0.02 * (16 - player_value)
                base_reward += penalty
            # -----------------------------------------------------------

            # 庄家行动阶段
            while True:
                dealer_value, has_usable_ace = self.calculate_hand_value(self.dealer_hand)
                if dealer_value > 17 or (dealer_value == 17 and not has_usable_ace):
                    break
                self.dealer_hand.append(self.deck.pop())

            # 最终点数计算
            player_value, _ = self.calculate_hand_value(self.player_hand)
            dealer_value, _ = self.calculate_hand_value(self.dealer_hand)

            # 胜负判定优先级修正
            if player_value > 21:
                final_reward = -1
            elif dealer_value > 21:
                final_reward = 1
            elif dealer_value > player_value:
                final_reward = -1
            elif player_value > dealer_value:
                final_reward = 1
            else:
                final_reward = 0

            # 合并基础奖励和最终结果
            total_reward = final_reward + base_reward

            return self.get_state(), total_reward, done