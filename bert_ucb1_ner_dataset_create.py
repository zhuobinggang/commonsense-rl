from bert_behavior_clone import Game_history_window_20
from command_generator import CommandGenerator
from functools import lru_cache
import bert_common
import common

def game_for_train(game_index):
    return bert_common.game_for_train(game_index, 1, 1, game_init_func=Game_ner)

@lru_cache(maxsize=None)
def get_command_generator():
    return CommandGenerator()



class Game_ner(Game_history_window_20):
    def available_actions_filtered_callback(self, available_actions):
        return available_actions
    def get_x(self): # 生成x的时候，用ner模型来生成，并提换掉原有的available_actions
        cg = get_command_generator()
        command_genererted = cg.commands_generate(self.env.info, need_taku_filter=True)
        self.available_actions = command_genererted
        return super().get_x()
