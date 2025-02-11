from typing import Any
from env import Env_extra_info
import common
import abstract_game_interface

class Env_twc_by_path(Env_extra_info):
    def __init__(self, game_path, no_augment = True) -> None:
        env = self.get_game_env_by_path(game_path)
        env.meta_info = f'{game_path}'
        env.meta_name = f'{game_path}'
        self.initiate_env(env)
        self.env = env
        self.no_augment = True;
        self.info = None


# NOTE: 2025.2.10 不使用room description，实验性质
def final_prompt_twc(room_name = '', action_history = '', action_list = ''):
    user_msg = ''
    user_msg += f"Room: {room_name if room_name else 'Unknown now'}\n"
    # user_msg += f"Recipe: {self.recipe if self.recipe else 'Unknown now'}\n"
    user_msg += f"Action history: {action_history if action_history else ''}\n" 
    user_msg += f'Available actions:\n{action_list}\n' if action_list else ''
    user_msg += 'Next action: [MASK]'
    return user_msg

class Game_interface(abstract_game_interface.Game_interface):
    def __init__(self, game_path, no_augment = True): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
         # train set
        self.env = Env_twc_by_path(game_path, no_augment)
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
        self.filename = f'TWC_{game_name}.json'
        self.won = False
        self.lost = False
        self.verbose = True # NOTE: 注意更改
        self.visited_dict = {} # 2024.12.21 用于存储访问过的地点次数
        self.desc_update_cache = {} # 2025.1.7 储存desc更新
        self.recipe = '' # 2025.1.13 储存菜谱
        self.init_hook()
    def init_hook(self):
        self.filter_startword_list = ['examine', 'close', 'eat', 'look', 'drop', 'inventory']
        self.walkthrough_with_meta_datas = [] # 2025.2.11 用于记录walkthrough
        self.walkthrough_path = self.game_path.replace('.ulx', '.walkthrough')
    def save_walkthrough(self):
        common.save_pickle_file(self.walkthrough_path, self.walkthrough_with_meta_datas)
    def load_walkthrough(self):
        return common.load_dataset(self.walkthrough_path)
    def get_walkthrough(self):
        if len(self.walkthrough_with_meta_datas) < 1:
            self.walkthrough_with_meta_datas = self.load_walkthrough()
        return [act for act, _ in self.walkthrough_with_meta_datas]
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        room_name = common.extract_room_name(description)
        action_history = common.action_obs_pairs_to_history(action_obs_pairs, seperator='>')
        action_list = common.actions_to_list_number(available_actions)
        usr = final_prompt_twc(room_name, action_history, action_list)
        self.x = usr
        sys = ''
        return sys, usr
    def print_walkthrough(self): # TODO: 打印时候要高亮得分项目
        for act, meta in self.walkthrough_with_meta_datas:
            txt = f'{act}' + (' [O]' if meta['score_up'] else '')
            print(txt)
    def action_obs_pairs_got_callback(self, action_obs_pairs):
        if not action_obs_pairs or len(action_obs_pairs) < 1:
            return
        act, obs = action_obs_pairs[-1]
        meta = {}
        if 'Your score has just gone up by one point' in obs:
            meta['score_up'] = True
        else:
            meta['score_up'] = False
        self.walkthrough_with_meta_datas.append((act, meta))
    def input(self, command):
        self.command = command
        command_idx = self.filtered_commands.index(command)
        self.finetune_triples.append((self.x, command_idx))
        self.act_and_output(command)
    def save_as_json(self):
        f = open(f'exp/auto_filename/{self.game_name}.jsonl', 'w')
        for x, y in self.finetune_triples:
            obj = {'x': x.strip(), 'y': str(y).strip()}
            f.write(str(obj) + '\n')
        f.close()