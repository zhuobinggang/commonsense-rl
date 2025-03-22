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
        # Added 2025.1.9
        self.action_templates = ''
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
        # Added 2025.1.9
        if not self.action_list:
            if self.action_templates:
                user_msg += f'Action templates:\n{self.action_templates}\n'
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

def prompt_from_game_classic(game):
    return prompt_from_env_feedback(game.description, game.inventory, game.available_actions, game.action_obs_pairs, another_room_info= '')


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
        self.verbose = False
        self.world_map = {} # 2024.12.21 用于存储访问过的地点次数
        self.desc_update_cache = {} # 2025.1.7 储存desc更新
    def construct_sys_usr(self):
        description, inventory, available_actions, action_obs_pairs = self.description, self.inventory, self.available_actions, self.action_obs_pairs
        sys, usr = prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, self.another_room_info)
        return sys, usr
    def move_command_succeeded_callback(self, action_obs):
        action, obs = action_obs
        obs = 'fake obs'
        return action, obs
    

class Ftwp_interface_by_path(human_play.Game_interface):
    def __init__(self, game_path, no_augment = True): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
         # train set
        self.env = Env_ftwp_by_path(game_path, no_augment)
        self.game_path = game_path
        game_name = game_path.split('/')[-1]
        self.game_name = game_name
        self.filename = f'FTWP_{game_name}.json'
        self.verbose = False
        self.init_hook()
        self.extras = None
    def init_all_params(self):
        self.dataset_index = 'trainset'
        self.hard_level_index = 'unknown'
        self.finetune_triples = [] # (sys, usr, command_next)
        self.current_sys = ''
        self.current_usr = ''
        self.command = ''
        self.updated_description = ''
        self.another_room_info = 'Unknown'
        self.won = False
        self.lost = False
        self.world_map = {} # 2024.12.21 用于存储访问过的地点次数
        self.desc_update_cache = {} # 2025.1.7 储存desc更新
        self.recipe = '' # 2025.1.13 储存菜谱
        self.filtered_commands = [] # 2025.2.11 用于使用指令代号来选择行动
        self.filter_startword_list = ['examine', 'put', 'close', 'insert', 'eat', 'look']
        self.kitchen_visited = False # 2025.2.28 用于判断是否访问过厨房
    def init_hook(self):
        pass
    def construct_sys_usr(self):
        description, inventory, available_actions, action_obs_pairs = self.description, self.inventory, self.available_actions, self.action_obs_pairs
        sys, usr = prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, self.another_room_info)
        return sys, usr
    def get_walkthrough(self):
        self.fetch_and_set_extras()
        return self.extras['walkthrough']
    def fetch_and_set_extras(self):
        if self.extras:
            return
        from ftwp_info import extra_info_by_game_path
        self.extras = extra_info_by_game_path(self.game_path)
    def get_score(self):
        return self.env.info['score']
    def get_max_score(self):
        return self.env.info['max_score']
    def get_inventory(self):
        return self.env.info['inventory']
    def get_inventory_as_set(self): # 2025.3.16
        return common.inventory_text_as_set(self.env.info['inventory'])
    def get_location(self):
        return common.extract_room_name(self.env.info['description'])
    def get_recipe(self):
        print('WARNING: get_recipe() will cheat, use it only if you understand.')
        self.fetch_and_set_extras()
        return self.extras['recipe']
    def available_actions_filtered_callback(self, filtered_commands):
        # return ['inventory'] + filtered_commands # NOTE: 这个会导致性能大幅下降
        return filtered_commands # NOTE: 2025.3.18 使用handicap1之后，不再需要inventory

def game_for_test():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0()
    game = Ftwp_interface_by_path(file_paths[1])
    game.verbose = True
    return game

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

# @history 2024.12.18 change dataset size from 10 to 20
def batch_training_file_prepare():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0(20) 
    for file_path in file_paths:
        game = Ftwp_interface_by_path(file_path)
        game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')

# ===================== 读取所有auto files，打乱然后合成out.jsonl =========================

def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare()

train_file_path = 'exp/ftwp_finetune_4omini_base/training.jsonl'

def valid_then_upload():
    from finetune_simplify_desc import valid, finetune_file_upload
    valid(train_file_path)
    finetune_file_upload(train_file_path)

FILE_ID_10_GAMES = 'file-PUSGGosbQA2HYyshhM9z2G'
FILE_ID_20_GAMES = 'file-3m7raJUnf3FtVgUx87coeJ'

def finetune_ftwp():
    from finetune_simplify_desc import finetune
    finetune(FILE_ID_20_GAMES)

TUNED_MODELS_10_GAMES = ['ft:gpt-4o-mini-2024-07-18:personal::AeIFBBWr:ckpt-step-143', 'ft:gpt-4o-mini-2024-07-18:personal::AeIFBkas:ckpt-step-286', 'ft:gpt-4o-mini-2024-07-18:personal::AeIFBgef']
BEST_10_GAMES_MODEL = TUNED_MODELS_10_GAMES[2]
TUNED_MODELS_20_GAMES = ['ft:gpt-4o-mini-2024-07-18:personal::Afh2XnJQ:ckpt-step-285', 'ft:gpt-4o-mini-2024-07-18:personal::Afh2X4xK:ckpt-step-570', 'ft:gpt-4o-mini-2024-07-18:personal::Afh2YySv']
BEST_20_GAMES_MODEL = TUNED_MODELS_20_GAMES[2]

# ==================== run valid & test ============================

def llm_auto_play(game, game_index, testing = True, file_prefix = 'B0_', max_try = 20, gpt_type = TUNED_MODELS_20_GAMES[0], sys_usr_from_game_func = prompt_from_game_classic):
    import finetuned_play
    return finetuned_play.llm_auto_play(game, game_index, testing, file_prefix, max_try, gpt_type, sys_usr_from_game_func)

def batch_valid():
    from ftwp_info import temp_test_valid_set
    _, valid_game_paths = temp_test_valid_set()
    for model in TUNED_MODELS_20_GAMES:
        for game_index, game_path in enumerate(valid_game_paths):
            game = Ftwp_interface_by_path(game_path)
            llm_auto_play(game, game_index, testing=False, gpt_type = model, max_try=20, sys_usr_from_game_func = prompt_from_game_classic)

# BEST = TUNED_MODELS_20_GAMES[2]

# @history: 2024.12.18 从valid集合中获取临时的test set
# @history: 2024.12.18 将训练集扩大到20重新实验
def run_test_temp():
    from ftwp_info import temp_test_valid_set
    test_game_paths, _ = temp_test_valid_set()
    model = BEST_20_GAMES_MODEL
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_interface_by_path(game_path)
        llm_auto_play(game, game_index, testing=True, gpt_type = model, max_try=20, sys_usr_from_game_func = prompt_from_game_classic)


def run_test():
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    model = BEST_20_GAMES_MODEL
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_interface_by_path(game_path)
        llm_auto_play(game, game_index, testing=True, gpt_type = model, max_try=20, sys_usr_from_game_func = prompt_from_game_classic)
