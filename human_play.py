# 我会玩耍5个测试游戏以创建训练集，目标是微调gpt4omini，看看效果如何。如果这个还不行，那可能就要放弃这个研究了。
import global_variable as G
import common

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

class Game_interface:
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
    def init_env(self, hard_level_index, game_index, dataset_index):
        from env import Env_extra_info
        return Env_extra_info(hard_level_index, game_index, dataset_index=dataset_index)
    def reset(self):
        self.act_and_output(None)
    def update_desciption(self, desc):
        return desc
    def desc_from_cache_or_update_desc(self, description):
        room_name = common.extract_room_name(description)
        # new key
        if room_name not in self.desc_update_cache:
            self.desc_update_cache[room_name] = {'desc': '', 'desc_updated': ''}
        # main logic
        if self.desc_update_cache[room_name]['desc'] == description: # 说明已经请求过了，直接返回cache
            updated_description = self.desc_update_cache[room_name]['desc_updated']
        else:
            updated_description = self.update_desciption(description)
            self.desc_update_cache[room_name]['desc_updated'] = updated_description
            self.desc_update_cache[room_name]['desc'] = description # 记得记录desc
        self.updated_description = updated_description
        return updated_description
    def is_move_command(self, command):
        if not command:
            return False
        return command.startswith('go ')
    def set_to_body(self, description, inventory, available_actions, action_obs_pairs):
        self.description = description
        self.inventory = inventory
        self.available_actions = available_actions
        self.action_obs_pairs = action_obs_pairs
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        sys, usr = prompt_from_env_feedback(description, inventory, available_actions, action_obs_pairs, self.another_room_info)
        return sys, usr
    def get_updated_description_before_description_update(self):
        return self.updated_description
    def available_actions_got_callback(self, available_actions):
        pass
    def move_command_succeeded_callback(self, action_obs):
        action, obs = action_obs
        room_name = common.extract_room_name(obs)
        if not room_name:
            print('XXXXXXXXXXXXXXX WRONG SITUATION XXXXXXXXXXXXXX')
            return obs
        if room_name not in self.visited_dict:
            self.visited_dict[room_name] = 0
        self.visited_dict[room_name] += 1
        # NOTE: Do not change obs as default behavior. Only save info into visited dict.
        # obs = obs + f' Visited {self.visited_dict[room_name]} times.'
        return action, obs
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
    def available_actions_filter(self, commands):
        if not commands:
            return
        word_list = ['examine', 'put', 'close', 'insert', 'eat', 'look']
        filtered_commands = []
        for command in commands:
            if not any(command.startswith(word) for word in word_list):
                filtered_commands.append(command)
        specific_commands = ['examine cookbook', 'eat meal']
        for command in specific_commands:
            if command in commands:
                filtered_commands.append(command)
        filtered_commands = ['inventory'] + filtered_commands
        return filtered_commands
    def act_and_output(self, command = None):
        description, inventory, available_actions, action_obs_pairs = self.env.act(command)
        available_actions = self.available_actions_filter(available_actions)
        self.save_if_checking_recipe(action_obs_pairs)
        self.available_actions_got_callback(available_actions)
        if description is not None:
            if self.is_move_command(command):
                action_obs_pairs[-1] = self.move_command_succeeded_callback(action_obs_pairs[-1])
            description = self.desc_from_cache_or_update_desc(description)
            self.set_to_body(description, inventory, available_actions, action_obs_pairs)
            sys, usr = self.construct_sys_usr(self.description, self.inventory, self.available_actions, self.action_obs_pairs)
            self.current_sys = sys
            self.current_usr = usr
            self.won = self.env.is_won(self.env.info)
            self.lost = self.env.is_lost(self.env.info)
        else:
            self.won = self.env.is_won(self.env.info)
            self.lost = self.env.is_lost(self.env.info)
        if self.verbose:
            print(self.current_sys + self.current_usr)
    def input(self, command):
        self.command = command
        self.finetune_triples.append((self.current_sys, self.current_usr, self.command))
        self.act_and_output(command)
    def save_as_json(self):
        f = open(f'exp/auto_filename/{self.filename}', 'w')
        for sys, usr, command in self.finetune_triples:
            obj = {'messages': [{"role": "system", "content": sys.strip()}, {"role": "user", "content": usr.strip()}, {"role": "assistant", "content": command.strip()}]}
            f.write(str(obj) + '\n')
        f.close()
    def output_actions(self):
        actions = [command for sys, usr, command in self.finetune_triples]
        return actions
    def save_readable(self):
        f = open(f'exp/auto_filename/{self.filename}.txt', 'w')
        for sys, usr, command in self.finetune_triples:
            f.write(sys + usr + '\n' + command + '\n\n')
        f.close()
    def auto_play(self, action_list):
        self.reset()
        for action in action_list:
            self.input(action)
    def get_score(self):
        return self.env.info['score']
    def get_max_score(self):
        return self.env.info['max_score']


def auto_play(game_index, action_list):
    game = Game_interface(game_index)
    game.reset()
    for action in action_list:
        game.input(action)
    return game