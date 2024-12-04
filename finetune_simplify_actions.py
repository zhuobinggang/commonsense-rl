from finetune_simplify import Game_simplify, game_played
import common
import global_variable as G


ACTION_TEMPLATES = """
Action templates:
* examine <container | supporter | object>
* open <container>
* close <container>
* take <object>
* put <object> on <supporter>
* put <object> in <container>
""".strip()

def action_template(movable_direction = ''):
    result = ''
    if not movable_direction:
        result = ACTION_TEMPLATES
    else:
        result = ACTION_TEMPLATES + f'\n* go {movable_direction}'
    # print(result)
    return result

class Promptor_simple_actions:
    def __init__(self):
        self.task = G.TASK_FINETUNE
        self.action_history = ''
        self.inventory = ''
        self.another_room_info = ''
        self.current_enviroment = ''
        self.movable_direction = '' # TODO: 需要填充
        # last consideration and action
        self.prev_consideration = ''
        self.prev_action = ''
        # others
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''

    def build(self):
        system_msg = ''
        system_msg += f'Task: {self.task}\n' if self.task else ''
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Action history: {self.action_history}\n' if self.action_history else ''
        user_msg += f'Inventory: {self.inventory}\n' if self.inventory else ''
        user_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        user_msg += f'Current environment: {self.current_enviroment}\n' if self.current_enviroment else ''
        user_msg += action_template(self.movable_direction).strip() + '\n'
        user_msg += 'Next action: '
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

def prompt_from_env_feedback_simple_actions(description, inventory, action_obs_pairs, another_room_info, movable_direction):
    promptor = Promptor_simple_actions()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.another_room_info = another_room_info
    promptor.movable_direction = movable_direction
    promptor.build()
    return promptor.system_msg, promptor.user_msg


def dic_to_sys_usr_func_simple_action_list(dic):
    return prompt_from_env_feedback_simple_actions(dic['description'], dic['inventory'], dic['action_obs_pairs'], dic['another_room_info'], dic['movable_direction'])

def game_played_and_save_training_file(game_index, output_file_path = 'exp/auto_filename/dd.jsonl'):
    import json
    game = game_played(game_index)
    prompt_necessary_info_dics = game.prompt_necessary_info_dics
    f = open(output_file_path, 'w')
    for dic in prompt_necessary_info_dics:
        sys, usr = dic_to_sys_usr_func_simple_action_list(dic) # NOTE: 使用action template而不是action list
        agent = dic['command']
        obj = common.training_line_prepare(sys, usr, agent)
        line = json.dumps(obj)
        f.write(line + '\n')
    f.close()

def batch_train_file_prepare():
    for game_index in range(5):
        game_played_and_save_training_file(game_index, output_file_path=f'exp/auto_filename/train_simplify_action_game{game_index}.jsonl')

# ================== randomize training file & output ==================

from finetune_simplify import lines_random_train_file_prepare

# ================== validation =================

training_data_path ='exp/finetune_4omini_simplify_actions/training.jsonl'
from finetune_simplify import valid, finetune_file_upload, finetune

def valid_then_upload():
    valid(training_data_path)
    finetune_file_upload(training_data_path)

training_file_id = 'file-AEGSMdGADzRqD8fz74EgZe'

def finetune_simple_actions():
    finetune(training_file_id)

MODELS = ['ft:gpt-4o-mini-2024-07-18:personal::AaEAigkN:ckpt-step-70',
          'ft:gpt-4o-mini-2024-07-18:personal::AaEAiVmA:ckpt-step-140',
          'ft:gpt-4o-mini-2024-07-18:personal::AaEAiAUu']

# ================== play =====================

from finetune_simplify import llm_auto_play

def sys_usr_from_game(game):
    return prompt_from_env_feedback_simple_actions(game.description, game.inventory, game.action_obs_pairs, game.another_room_info, game.movable_direction)

# TODO: 测试　
llm_auto_play(0, testing=False, gpt_type = MODELS[0], max_try=10, sys_usr_from_game_func = sys_usr_from_game)