from ftwp_finetune_simplify_desc import Promptor_ftwp, Ftwp_interface_by_path, game_played_and_save_training_file
import common
import finetuned_play

ACTION_TEMPLATES = """
* inventory
* examine <object>
* open <object>
* take <object> from <object>
* cook <object> with <stove | oven>
* drop <object>
* <slice | dice | chop> <object> with knife
* prepare meal
* eat meal
* go <direction>
""".strip()


def prompt_from_env_feedback_action_template(description, inventory, action_obs_pairs):
    promptor = Promptor_ftwp()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.action_templates = ACTION_TEMPLATES # NOTE: Only important
    promptor.build()
    return promptor.system_msg, promptor.user_msg


def prompt_from_game_action_template(game):
    return prompt_from_env_feedback_action_template(game.description, game.inventory, game.action_obs_pairs)

class Ftwp_simple_actions(Ftwp_interface_by_path):
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        sys, usr = prompt_from_env_feedback_action_template(description, inventory, action_obs_pairs)
        return sys, usr
    

def test_template():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0(20) 
    file_path = file_paths[0]
    game = Ftwp_simple_actions(file_path)
    game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')


# @history 2024.1.9 create
def batch_training_file_prepare():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0(20) 
    for file_path in file_paths:
        game = Ftwp_simple_actions(file_path)
        game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')

def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare(out_path='exp/auto_filename/ftwp_simplify_actions.jsonl')


train_file_path = 'exp/auto_filename/ftwp_simplify_actions.jsonl'

def valid_then_upload():
    from finetune_simplify_desc import valid, finetune_file_upload
    valid(train_file_path)
    return finetune_file_upload(train_file_path)

FILE_ID = 'file-Xfaxitkoh2nPjftUgBUQm3'

def finetune_ftwp_simplify_actions():
    from finetune_simplify_desc import finetune
    finetune(FILE_ID)


MODELS = ['ft:gpt-4o-mini-2024-07-18:personal::AnhBLsc7:ckpt-step-285', 'ft:gpt-4o-mini-2024-07-18:personal::AnhBLv3s:ckpt-step-570', 'ft:gpt-4o-mini-2024-07-18:personal::AnhBMZL3']
BEST = MODELS[2]

def llm_auto_play_action_selector(game, game_index, testing = True, file_prefix = 'B0_', max_try = 20, gpt_type = MODELS[0], sys_usr_from_game_func = prompt_from_game_action_template):
    return finetuned_play.llm_auto_play_action_selector(game, game_index, testing, file_prefix, max_try, gpt_type, sys_usr_from_game_func)


def batch_valid():
    from ftwp_info import temp_test_valid_set
    _, valid_game_paths = temp_test_valid_set()
    for model in MODELS:
        for game_index, game_path in enumerate(valid_game_paths):
            game = Ftwp_simple_actions(game_path)
            llm_auto_play_action_selector(game, game_index, testing=False, gpt_type = model, max_try=20, sys_usr_from_game_func = prompt_from_game_action_template)

# @history: 2024.12.18 从valid集合中获取临时的test set
# @history: 2024.12.18 将训练集扩大到20重新实验
def run_test_temp():
    from ftwp_info import temp_test_valid_set
    test_game_paths, _ = temp_test_valid_set()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_simple_actions(game_path)
        llm_auto_play_action_selector(game, game_index, testing=True, gpt_type = model, max_try=20, sys_usr_from_game_func = prompt_from_game_action_template)


def run_test():
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_simple_actions(game_path)
        llm_auto_play_action_selector(game, game_index, testing=True, gpt_type = model, max_try=20, sys_usr_from_game_func = prompt_from_game_action_template)