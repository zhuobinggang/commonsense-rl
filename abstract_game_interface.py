import common

class Game_interface:
    def __init__(self, no_augment = True): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
         # train set
        self.env = self.get_env()
        self.game_name = 'default'
        self.dataset_index = 'trainset'
        self.hard_level_index = 'unknown'
        self.finetune_triples = [] # (sys, usr, command_next)
        self.current_sys = ''
        self.current_usr = ''
        self.command = ''
        self.updated_description = ''
        self.another_room_info = 'Unknown'
        self.filename = f'TWC_{self.game_name}.json'
        self.verbose = True # NOTE: 注意更改
        self.visited_dict = {} # 2024.12.21 用于存储访问过的地点次数
        self.desc_update_cache = {} # 2025.1.7 储存desc更新
        self.recipe = '' # 2025.1.13 储存菜谱
        self.init_filter_startword_list()
        self.init_hook()
        self.filtered_commands = [] # 2025.2.11 用于使用指令代号来选择行动
    def __call__(self, command_text_or_idx):
        if isinstance(command_text_or_idx, str):
            self.input(command_text_or_idx)
        elif isinstance(command_text_or_idx, int):
            command = self.filtered_commands[command_text_or_idx]
            self.input(command)
    def get_env(self):
        raise Exception('需要重新实现这个函数！')
    def init_filter_startword_list(self):
        self.filter_startword_list = ['examine', 'close', 'eat', 'look', 'drop', 'inventory']
    def init_hook(self):
        pass
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
    def fake_prompt(self, room_name = '', action_history = '', action_list = ''):
        user_msg = ''
        user_msg += f"Room: {room_name if room_name else 'Unknown now'}\n"
        # user_msg += f"Recipe: {self.recipe if self.recipe else 'Unknown now'}\n"
        user_msg += f"Action history: {action_history if action_history else ''}\n" 
        user_msg += f'Available actions:\n{action_list}\n' if action_list else ''
        user_msg += 'Next action: [MASK]'
        return user_msg
    def construct_sys_usr(self, description, inventory, available_actions, action_obs_pairs):
        room_name = common.extract_room_name(description)
        action_history = common.action_obs_pairs_to_history(action_obs_pairs, seperator='>')
        action_list = common.actions_to_list_number(available_actions)
        usr = self.fake_prompt(room_name, action_history, action_list)
        sys = ''
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
    def get_obs(self, act_obs):
        if not act_obs or len(act_obs) < 1:
            return ''
        action, obs = act_obs[-1]
        return obs
    def available_actions_filtered_callback(self, filtered_commands):
        return filtered_commands
    def available_actions_filter(self, commands):
        if not commands:
            return
        word_list = self.filter_startword_list
        filtered_commands = []
        for command in commands:
            if not any(command.startswith(word) for word in word_list):
                filtered_commands.append(command)
        specific_commands = ['examine cookbook', 'eat meal']
        for command in specific_commands:
            if command in commands:
                filtered_commands.append(command)
        # filtered_commands = ['inventory'] + filtered_commands
        filtered_commands = self.available_actions_filtered_callback(filtered_commands)
        self.filtered_commands = filtered_commands
        return filtered_commands
    def action_obs_pairs_got_callback(self, action_obs_pairs):
        pass
    def is_won(self):
        return self.env.is_won(self.env.info)
    def is_lost(self):
        return self.env.is_lost(self.env.info)
    def act_and_output(self, command = None):
        if command and (self.is_won() or self.is_lost()):
            print('不需要继续，请退出')
            return
        description, inventory, available_actions, action_obs_pairs = self.env.act(command)
        available_actions = self.available_actions_filter(available_actions)
        self.action_obs_pairs_got_callback(action_obs_pairs)
        self.available_actions_got_callback(available_actions)
        if description is not None: # 任何时候都不会是NONE
            if self.is_move_command(command):
                action_obs_pairs[-1] = self.move_command_succeeded_callback(action_obs_pairs[-1])
            description = self.desc_from_cache_or_update_desc(description)
            self.set_to_body(description, inventory, available_actions, action_obs_pairs)
            sys, usr = self.construct_sys_usr(self.description, self.inventory, self.available_actions, self.action_obs_pairs)
            self.current_sys = sys
            self.current_usr = usr
        else:
            raise Exception('更改env逻辑之后，现在不可能落入这个判断')
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
    def back(self):
        actions = self.output_actions()
        actions = actions[:-1]
        self.reset()
        self.auto_play(actions)
    def print_walkthrough(self):
        return self.output_actions()
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
        return self.env.info['score'][0]
    def get_max_score(self):
        return self.env.info['max_score'][0]
    


class Game_state:
    def __init__(self) -> None:
        self.x = ''
        self.action_list = []