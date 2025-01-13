from ftwp_finetune_simplify_desc import Ftwp_interface_by_path, game_played_and_save_training_file
from ftwp_finetune_simplify_actions import prompt_from_env_feedback_action_template, llm_auto_play_action_selector
from llm_simplify import quest_4omini_simplify_desc

class Ftwp_simple_both(Ftwp_interface_by_path):
    def update_desciption(self, description):
        return quest_4omini_simplify_desc(description, need_prompt=False)
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        sys, usr = prompt_from_env_feedback_action_template(description, inventory, action_obs_pairs)
        return sys, usr
    
def batch_training_file_prepare():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0(20) 
    for file_path in file_paths:
        game = Ftwp_simple_both(file_path)
        game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')

def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare(out_path='exp/auto_filename/ftwp_simplify_desc.jsonl')

train_file_path = 'exp/auto_filename/ftwp_simplify_desc.jsonl'

def valid_then_upload():
    from finetune_simplify_desc import valid, finetune_file_upload
    valid(train_file_path)
    return finetune_file_upload(train_file_path)

FILE_ID = 'file-HHZEgaSSogdywxGC7D9trE'


def finetune_ftwp_simplify():
    from finetune_simplify_desc import finetune
    finetune(FILE_ID)

MODELS = ['ft:gpt-4o-mini-2024-07-18:personal::Ao38CKhH:ckpt-step-285', 'ft:gpt-4o-mini-2024-07-18:personal::Ao38CFxR:ckpt-step-570', 'ft:gpt-4o-mini-2024-07-18:personal::Ao38DneU']
BEST = MODELS[2]

def batch_valid():
    from ftwp_info import temp_test_valid_set
    _, valid_game_paths = temp_test_valid_set()
    for model in MODELS:
        for game_index, game_path in enumerate(valid_game_paths):
            game = Ftwp_simple_both(game_path)
            llm_auto_play_action_selector(game, game_index, testing=False, gpt_type = model, max_try=20)


# @history: 2024.12.18 从valid集合中获取临时的test set
# @history: 2024.12.18 将训练集扩大到20重新实验
def run_test_temp():
    from ftwp_info import temp_test_valid_set
    test_game_paths, _ = temp_test_valid_set()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_simple_both(game_path)
        llm_auto_play_action_selector(game, game_index, testing=True, gpt_type = model, max_try=20)


def run_test():
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_simple_both(game_path)
        llm_auto_play_action_selector(game, game_index, testing=True, gpt_type = model, max_try=20)