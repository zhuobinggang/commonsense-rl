# 我会玩耍5个测试游戏以创建训练集，目标是微调gpt4omini，看看效果如何。如果这个还不行，那可能就要放弃这个研究了。
import global_variable as G
import common
import abstract_game_interface

class Prompt_builder_for_finetune:
    def __init__(self):
        self.task = G.TASK_FINETUNE
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
        user_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        user_msg += f'Current environment: {self.current_enviroment}\n' if self.current_enviroment else ''
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += 'Next action: '
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

def prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, another_room_info):
    promptor = Prompt_builder_for_finetune()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.action_list = common.actions_to_list(available_actions)
    promptor.another_room_info = another_room_info
    promptor.build()
    return promptor.system_msg, promptor.user_msg

class Game_interface(abstract_game_interface.Game_interface):
    def __init__(self, game_index, dataset_index = 0, hard_level_index = 2): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
         # train set
        self.env = self.init_env(hard_level_index, game_index, dataset_index)
        self.game_index = game_index
        self.dataset_index = dataset_index
        self.hard_level_index = hard_level_index
        self.finetune_triples = [] # (sys, usr, agent)
        self.current_sys = ''
        self.current_usr = ''
        self.command = ''
        self.updated_description = ''
        self.another_room_info = 'Unknown'
        self.filename = f'finetune_dataset_{dataset_index}_level_{hard_level_index}_game_{game_index}.json'
        self.won = False
        self.lost = False
        self.verbose = False
        self.visited_dict = {} # 2024.12.21 用于存储访问过的地点次数
        self.desc_update_cache = {} # 2025.1.7 储存desc更新
        self.recipe = '' # 2025.1.13 储存菜谱
        self.filter_startword_list = ['examine', 'put', 'close', 'insert', 'eat', 'look']
    def init_env(self, hard_level_index, game_index, dataset_index):
        from env import Env_extra_info
        return Env_extra_info(hard_level_index, game_index, dataset_index=dataset_index)
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        sys, usr = prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, self.another_room_info)
        return sys, usr
    def save_if_checking_recipe(self, act_obs):
        if not act_obs or len(act_obs) < 1:
            return
        if self.recipe != '': # 说明已经保存过了
            return
        action, obs = act_obs[-1]
        if action != 'examine cookbook':
            return
        if not common.is_recipe_feedback(obs):
            return
        # print('检查菜谱成功，抽取菜谱内容')
        act_obs[-1] = (action, 'Recipe got!')
        self.recipe = common.extract_recipe(obs)
    def action_obs_pairs_got_callback(self, action_obs_pairs):
        self.save_if_checking_recipe(action_obs_pairs) 


def auto_play(game_index, action_list):
    game = Game_interface(game_index)
    game.reset()
    for action in action_list:
        game.input(action)
    return game