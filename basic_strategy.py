# basic_strategy.py
class BasicStrategy:
    @staticmethod
    def get_action(state):
        """
        输入: (player_total, dealer_showing, usable_ace)
        返回: 0（停牌）或 1（要牌）
        策略参考自经典 Blackjack 基本策略表
        """
        player_total, dealer_card, usable_ace = state

        # 针对软手（有可用A），即总点数实际上是软总和
        if usable_ace:
            # 注意：这里的 player_total 是经过调整的总点数（A可作为11），因此可以直接判断
            if player_total <= 17:
                # 软手 13~17 均建议要牌
                return 1
            elif player_total == 18:
                # 对软18，根据庄家的牌面做决策
                # 如果庄家显示2,7,8建议停牌，其它情况下选择要牌（注意：部分策略也建议转为双倍下注）
                if dealer_card in [2, 7, 8]:
                    return 0
                else:
                    return 1
            else:
                # 软 19 以上通常为停牌
                return 0

        # 针对硬手（无可用A）：
        else:
            if player_total <= 11:
                return 1
            elif player_total >= 17:
                return 0
            else:
                # player_total 在 12～16之间，依据庄家亮牌来决策
                # 若庄家亮牌在 2~6，建议停牌；否则选择要牌
                if dealer_card in [2, 3, 4, 5, 6]:
                    return 0
                else:
                    return 1
