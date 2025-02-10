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
    def init_hook(self):
        pass
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
        filename = self.filename + f'score{self.get_score()}_of_{self.get_max_score()}'
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
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1) # TODO: 2025.1.20 为了能够对应强化学习，应该将argmax改为sampling
    command_str = tokenizer.decode(predicted_token_id).strip()
    # print(f'Command IDX: {command_str}')
    command_index = int(command_str)
    command = game.available_actions[command_index]
    # print(f'Command: {command}')
    return command

# @history: 1.17: 需要分开彻底分开
def valid_paths():
    from ftwp_info import all_valid_game_paths
    return all_valid_game_paths(shuffle = True)[-30:]

def batch_test(model, tokenizer = None, save_readable = True, test_game_paths = [], file_prefix = ''):
    if not tokenizer:
        tokenizer = default_tokenizer()
    if len(test_game_paths) < 1:
        test_game_paths = valid_paths()
    scores = []
    max_scores = []
    for path in test_game_paths:
        game = Game_for_bert(path)
        game.verbose = False
        game.filename = file_prefix + game.filename # Added 2025.2.8
        score, max_score = trained_model_autoplay(game, model, tokenizer, save_readable)
        scores.append(score)
        max_scores.append(max_score)
    return sum(scores) / sum(max_scores)

def batch_valid(model, tokenizer = None, save_readable = True):
    return batch_test(model, tokenizer, save_readable, test_game_paths=valid_paths())
    
# ================= compare llm performance ===================

# @history: 2024.1.17
def run_test_temp(model):
    from ftwp_info import temp_test_valid_set
    test_game_paths, _ = temp_test_valid_set()
    return batch_test(model, save_readable=True, test_game_paths=test_game_paths)


def run_test(model):
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    return batch_test(model, save_readable=True, test_game_paths=test_game_paths)

def final_test():
    model_paths = ['/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart0.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart1.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart2.tch']
    results = [] # 3 models, 2 test methods
    for model_path in model_paths:
        model, toker = load_trained_model(model_path)
        temp_score = run_test_temp(model)
        real_score = run_test(model)
        results.append((temp_score, real_score))
    return results

def run_test_full(model, file_prefix = ''):
    from ftwp_info import all_test_game_paths
    test_game_paths=  all_test_game_paths()
    return batch_test(model, save_readable=True, test_game_paths=test_game_paths, file_prefix=file_prefix)

def run_test_full_with_model():
    import numpy as np
    model_paths = ['/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart0.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart1.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart2.tch']
    results = [] # 3 models, 2 test methods
    logger = common.Logger_simple(file_name='run_test_full_with_model_log')
    for model_idx, model_path in enumerate(model_paths):
        model, toker = load_trained_model(model_path)
        logger.add(f'Model {model_idx} testing...')
        score = run_test_full(model, file_prefix=f'M{model_idx}')
        logger.add(f'Model {model_idx} tested! Score = {score}.')
        logger.write_txt_log()
        results.append(score)
    logger.add(f'All model tested! Average score = {np.mean(results)}')
    logger.write_txt_log()
    return results