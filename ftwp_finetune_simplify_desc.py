from interface_ftwp import Ftwp_interface_by_path, Promptor_ftwp
from llm_simplify import quest_4omini_simplify_desc
from interface_ftwp import llm_auto_play
import common

LEGACY_VARS = {
    'models_before_simplification_prompt_reconstruction': ['ft:gpt-4o-mini-2024-07-18:personal::AmcYzhuG:ckpt-step-285', 'ft:gpt-4o-mini-2024-07-18:personal::AmcYzEqj:ckpt-step-570', 'ft:gpt-4o-mini-2024-07-18:personal::AmcYzgWT'],
    'model_best_before_simplification_prompt_reconstruction': 'ft:gpt-4o-mini-2024-07-18:personal::AmcYzgWT',
    'train_file_id_before_simplification_prompt_reconstruction': 'file-VtcfFgN4WM1cvsArgykDd5'
}

# =================== Promptor ===================

def prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs):
    promptor = Promptor_ftwp()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.action_list = common.actions_to_list(available_actions)
    promptor.build()
    return promptor.system_msg, promptor.user_msg

# ======================================

class Ftwp_simple_desc(Ftwp_interface_by_path):
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        sys, usr = prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs)
        return sys, usr
    def update_desciption(self, description):
        return quest_4omini_simplify_desc(description, need_prompt=False)

def game_played_and_save_training_file(game, output_file_path):
    import json
    game.verbose = False
    game.auto_play(game.get_walkthrough()) # TODO: 修改game游玩时生成的prompt
    f = open(output_file_path, 'w')
    for sys, usr, agent in game.finetune_triples:
        obj = common.training_line_prepare(sys, usr, agent)
        line = json.dumps(obj)
        f.write(line + '\n')
    f.close()

# @history 2024.1.6 create
def batch_training_file_prepare():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0(20) 
    for file_path in file_paths:
        game = Ftwp_simple_desc(file_path)
        game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')

def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare(out_path='exp/auto_filename/ftwp_simplify_desc.jsonl')


train_file_path = 'exp/auto_filename/ftwp_simplify_desc.jsonl'

def valid_then_upload():
    from finetune_simplify_desc import valid, finetune_file_upload
    valid(train_file_path)
    return finetune_file_upload(train_file_path)

FILE_ID = 'file-RK4rADgtGgZqXKDFpzxy8W'

def finetune_ftwp_simplify():
    from finetune_simplify_desc import finetune
    finetune(FILE_ID)

MODELS = ['ft:gpt-4o-mini-2024-07-18:personal::AmyTTM95:ckpt-step-285', 'ft:gpt-4o-mini-2024-07-18:personal::AmyTTzDW:ckpt-step-570', 'ft:gpt-4o-mini-2024-07-18:personal::AmyTUwF1']
BEST = MODELS[2]

def batch_valid():
    from ftwp_info import temp_test_valid_set
    _, valid_game_paths = temp_test_valid_set()
    for model in MODELS:
        for game_index, game_path in enumerate(valid_game_paths):
            game = Ftwp_simple_desc(game_path)
            llm_auto_play(game, game_index, testing=False, gpt_type = model, max_try=20)

# @result: 0.6209150326797386
def run_test_temp():
    from ftwp_info import temp_test_valid_set
    test_game_paths, _ = temp_test_valid_set()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_simple_desc(game_path)
        llm_auto_play(game, game_index, testing=True, gpt_type = model, max_try=20)

# @result: 0.3230769230769231
def run_test():
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_simple_desc(game_path)
        llm_auto_play(game, game_index, testing=True, gpt_type = model, max_try=20)