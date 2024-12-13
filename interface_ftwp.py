# Game interface for FTWP games
import human_play
from env_for_ftwp import Env_ftwp, Env_ftwp_by_path
import global_variable as G
import common

# ============================= Promptor ================================

class Promptor_ftwp:
    def __init__(self):
        self.task = G.TASK_FTWP
        self.action_history = ''
        self.inventory = ''
        self.another_room_info = ''
        self.current_enviroment = ''
        self.action_list = ''
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
        # NOTE: 感觉不需要另一个房间的信息，在FTWP环境下
        # user_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        user_msg += f'Current environment: {self.current_enviroment}\n' if self.current_enviroment else ''
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += 'Next action (answer with the command directly): '
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

def prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, another_room_info):
    promptor = Promptor_ftwp()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.action_list = common.actions_to_list(available_actions)
    promptor.another_room_info = another_room_info
    promptor.build()
    return promptor.system_msg, promptor.user_msg


# ============================= Interface ===============================

class Ftwp_interface(human_play.Game_interface):
    def __init__(self, recipe_num = 1, game_index = 0, no_augment = True): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
         # train set
        self.env = Env_ftwp(recipe_num=recipe_num, game_index=game_index, no_augment=no_augment)
        self.game_index = game_index
        self.dataset_index = 'trainset'
        self.hard_level_index = 'unknown'
        self.finetune_triples = [] # (sys, usr, command_next)
        self.current_sys = ''
        self.current_usr = ''
        self.command = ''
        self.updated_description = ''
        self.another_room_info = 'Unknown'
        self.filename = f'FTWP_{self.dataset_index}_level_{self.hard_level_index}_game_{self.game_index}.json'
        self.won = False
        self.verbose = True
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        sys, usr = prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, self.another_room_info)
        return sys, usr
    

class Ftwp_interface_by_path(human_play.Game_interface):
    def __init__(self, game_path, no_augment = True): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
         # train set
        self.env = Env_ftwp_by_path(game_path, no_augment)
        self.game_path = game_path
        game_name = game_path.split('/')[-1]
        self.game_name = game_name
        self.dataset_index = 'trainset'
        self.hard_level_index = 'unknown'
        self.finetune_triples = [] # (sys, usr, command_next)
        self.current_sys = ''
        self.current_usr = ''
        self.command = ''
        self.updated_description = ''
        self.another_room_info = 'Unknown'
        self.filename = f'FTWP_{game_name}.json'
        self.won = False
        self.verbose = True
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        sys, usr = prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, self.another_room_info)
        return sys, usr
    def get_walkthrough(self):
        from ftwp_info import walkthrougn_by_game_path
        return walkthrougn_by_game_path(self.game_path)

def test_get_train_set():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0()
    return Ftwp_interface_by_path(file_paths[0])

def test_get_test_set():
    from ftwp_info import test_set_v0
    file_paths = test_set_v0()
    return Ftwp_interface_by_path(file_paths[0])

# ====================== training file prepare =========================

def game_played_and_save_training_file(game, output_file_path = 'exp/auto_filename/dd.jsonl'):
    import json
    game.verbose = False
    game.auto_play(game.get_walkthrough())
    f = open(output_file_path, 'w')
    for sys, usr, agent in game.finetune_triples:
        obj = common.training_line_prepare(sys, usr, agent)
        line = json.dumps(obj)
        f.write(line + '\n')
    f.close()