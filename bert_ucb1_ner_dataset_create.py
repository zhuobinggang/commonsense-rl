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
    def available_actions_filtered_callback(self, filtered_commands):
        return filtered_commands
    def filter_by_head_word(self, commands, head_words):
        word_list = head_words
        filtered_commands = []
        for command in commands:
            if not any(command.startswith(word) for word in word_list):
                filtered_commands.append(command)
        specific_commands = []
        if 'examine' in word_list:
            specific_commands.append('examine cookbook')
        if 'eat' in word_list:
            specific_commands.append('eat meal')
        for command in specific_commands:
            if command in commands:
                filtered_commands.append(command)
        return filtered_commands
    def available_actions_filter(self, commands, generated = False, inventory = ''): 
        if not generated: # NOTE: 2025.3.26 do not use available actions from meta data
            return []
        commands = self.filter_by_head_word(commands, common.FILTER_STARTWORD_LIST) # 先过滤一次common.FILTER_STARTWORD_LIST
        # 如果inventory中有knife，检查所有slice和chop指令，将物品名提取出来，检查是否在inventory中，如果不在，过滤掉
        return commands
    def get_x(self):
        cg = get_command_generator()
        command_genererted = cg.commands_generate(self.env.info)
        inventory = common.handle_inventory_text(self.env.info['inventory'])
        self.filtered_commands = self.available_actions_filter(command_genererted, generated = True, inventory = inventory)
        return super().get_x()


