# 使用ucb1来增强behavior cloning
# TODO: 将Game继承过来，增加get_state方法，获取游戏的inventory，recipe还有location信息组成一个state文本返回
# TODO: 将Model继承过来，增加get_command_logits(game_state)和next_action_ucb1(self, logits, state_action_count)方法
from bert_behavior_clone import Model_behavior_clone
from bert_common import Game_for_rl, get_command_logits_simple

class Game_with_state(Game_for_rl):
    def get_ucb1_state(self):
        inventory = self.get_inventory()
        recipe = self.get_recipe()
        location = self.get_location()
        state_text = f"Inventory: {inventory}\nRecipe: {recipe}\nLocation: {location}"
        return state_text

class Model_with_ucb1(Model_behavior_clone):
    def __init__(self, bert=None):
        super().__init__(bert)
        self.state_action_count = {}

    def get_command_logits(self, game_state):
        logits = get_command_logits_simple(self.bert, game_state)
        return logits

    def next_action_ucb1(self, logits, state_action_count):
        import numpy as np
        total_count = sum(state_action_count.values())
        ucb_values = {}
        for action, count in state_action_count.items():
            ucb_values[action] = logits[action] + np.sqrt(2 * np.log(total_count + 1) / (count + 1))
        best_action = max(ucb_values, key=ucb_values.get)
        return best_action