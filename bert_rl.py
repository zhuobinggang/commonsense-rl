# BERT + reinforcement learning for FTWP dataset.
# 先这样：只保留房间名，行动历史，菜谱，命令列表，然后用4000多个游戏进行行为克隆
import common
from interface_ftwp import game_for_test, Ftwp_interface_by_path



def game_for_test():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0()
    game = Game_for_bert(file_paths[1])
    game.verbose = True
    return game

class Prompt_builder_for_bert:
    def __init__(self):
        self.room_name = ''
        self.action_history = ''
        self.action_list = ''
        # others
        self.recipe = ''
        self.prompt = ''

    def build(self):
        user_msg = ''
        user_msg += f"Room: {self.room_name if self.room_name else 'Unknown now'}\n"
        user_msg += f"Recipe: {self.recipe if self.recipe else 'Unknown now'}\n"
        user_msg += f"Action history: {self.action_history if self.action_history else ''}\n" 
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += 'Next action: [MASK]'
        self.prompt = user_msg

def prompt_from_env_feedback_action_template(description, action_obs_pairs, available_actions, recipe):
    promptor = Prompt_builder_for_bert()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs, seperator='>')
    promptor.room_name = common.extract_room_name(description)
    promptor.action_list = common.actions_to_list_number(available_actions)
    promptor.recipe = recipe
    promptor.build()
    return promptor.prompt

def bert_prompt_from_game(game):
    return prompt_from_env_feedback_action_template(game.description, game.action_obs_pairs, game.available_actions, game.recipe)

def get_input_text(game):
    pass


class Game_for_bert(Ftwp_interface_by_path):
    def input(self, command):
        self.command = command
        try:
            x = bert_prompt_from_game(self)
            y = self.available_actions.index(command)
            self.finetune_triples.append((x, y)) # save x and y
            self.act_and_output(command)
        except:
            print(f'指令不存在，不需要执行: {command}')
            print(self.available_actions)
            print(self.filename)
    def save_as_json(self):
        f = open(f'exp/auto_filename/{self.filename}', 'w')
        for x, y in self.finetune_triples:
            obj = {'x': x.strip(), 'y': str(y).strip()}
            f.write(str(obj) + '\n')
        f.close()
    def output_actions(self):
        actions = [y for x, y in self.finetune_triples]
        return actions
    def save_readable(self):
        f = open(f'exp/auto_filename/{self.filename}.txt', 'w')
        for sys, usr, command in self.finetune_triples:
            f.write(sys + usr + '\n' + command + '\n\n')
        f.close()
    def auto_play(self, action_list):
        self.reset()
        for action in action_list:
            self.input(action)

# DONE: TEST
def test_get_recipe():
    game = game_for_test()
    game.verbose = False
    game.reset()
    game.input('go west')
    game.input('go south')
    game.input('examine cookbook')
    # print(game.recipe)
    print(bert_prompt_from_game(game))
    return game


def prepare_bert():
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

def game_played_and_save_training_file(game, output_file_path):
    import json
    game.verbose = False
    game.auto_play(game.get_walkthrough()) # TODO: 修改game游玩时生成的prompt
    f = open(output_file_path, 'w')
    for x, y in game.finetune_triples:
        obj = {'x': x.strip(), 'y': str(y).strip()}
        line = json.dumps(obj)
        f.write(line + '\n')
    f.close()

def batch_training_file_prepare():
    from ftwp_info import all_train_game_paths
    file_paths = all_train_game_paths() 
    bad_train_files = []
    for file_path in file_paths:
        game = Game_for_bert(file_path)
        game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')
        if not game.won:
            print(f'有问题啊朋友，文件数量: {len(bad_train_files)}')
            bad_train_files.append(file_path)
    return bad_train_files


def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare(directory_path = '/home/taku/Downloads/cog2019_ftwp/procceeded_training_files/bert_trains', out_path='/home/taku/Downloads/cog2019_ftwp/procceeded_training_files/bert_trains.jsonl')

