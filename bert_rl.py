# BERT + reinforcement learning for FTWP dataset.
# 先这样：只保留房间名，行动历史，菜谱，命令列表，然后用4000多个游戏进行行为克隆
import common
from interface_ftwp import game_for_test, Ftwp_interface_by_path
from functools import lru_cache


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
        filename = self.filename + f'{self.get_score()}_of_{self.get_max_score()}'
        f = open(f'exp/auto_filename/{filename}.txt', 'w')
        for x, y in self.finetune_triples:
            f.write(f'{x}\n\n{y}\n\n')
        f.write(f'{self.get_score()}/{self.get_max_score()}')
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

# ==== auto play ========

def load_trained_model(path):
    from transformers import AutoTokenizer, ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBertForMaskedLM.from_pretrained(path)
    return model, tokenizer

@lru_cache(maxsize=None)
def default_tokenizer():
    from transformers import AutoTokenizer
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def model_for_test():
    return load_trained_model('/home/taku/Downloads/cog2019_ftwp/procceeded_training_files/bert_trained_0115.tch')

def trained_model_autoplay(game, model, tokenizer, save_readable = True):
    model.eval()
    game.reset()
    counter = 0
    while not any([counter >= 30, game.won, game.lost]):
        try:
            command = get_next_command(game, tokenizer, model)
            game.input(command)
            counter += 1
        except:
            print(f'根据BERT获取指令出问题了，返回分数0即可')
            return 0, 0
    if save_readable:
        game.save_readable()
    return game.get_score(), game.get_max_score()
    # logits = model(**inputs).logits
    # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

def get_next_command(game, tokenizer, model):
    import torch
    device = model.device
    x = bert_prompt_from_game(game)
    inputs = tokenizer(x, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    command_str = tokenizer.decode(predicted_token_id).strip()
    # print(f'Command IDX: {command_str}')
    command_index = int(command_str)
    command = game.available_actions[command_index]
    # print(f'Command: {command}')
    return command

def valid_paths():
    from ftwp_info import all_valid_game_paths
    return all_valid_game_paths(shuffle = True)[:20]

def batch_valid(model, tokenizer = None, save_readable = True):
    if not tokenizer:
        tokenizer = default_tokenizer()
    scores = []
    max_scores = []
    for path in valid_paths():
        game = Game_for_bert(path)
        game.verbose = False
        score, max_score = trained_model_autoplay(game, model, tokenizer, save_readable)
        scores.append(score)
        max_scores.append(max_score)
    return sum(scores) / sum(max_scores)
    
