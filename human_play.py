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
    promptor.another_room_info = another_room_info# TODO
    promptor.build()
    return promptor.system_msg, promptor.user_msg

class Game_interface:
    def __init__(self, game_index, dataset_index = 0, hard_level_index = 2): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
        from env import Env_extra_info
         # train set
        self.env = Env_extra_info(hard_level_index, game_index, dataset_index=dataset_index)
        self.finetune_triples = [] # (sys, usr, command_next)
        self.current_sys = ''
        self.current_usr = ''
        self.command = ''
        self.updated_description = ''
        self.another_room_info = 'Unknown'
        self.filename = f'finetune_dataset_{dataset_index}_level_{hard_level_index}_game_{game_index}.json'
        self.won = False
    def reset(self):
        self.act_and_output(None)
    def update_description(self, description):
        # TODO: 使用gpt4o来summarization
        return description
    def is_move_command(self, command):
        if not command:
            return False
        return command.startswith('go ')
    def set_to_body(self, description, inventory, available_actions, action_obs_pairs):
        self.description = description
        self.inventory = inventory
        self.available_actions = available_actions
        self.action_obs_pairs = action_obs_pairs
    def act_and_output(self, command = None):
        description, inventory, available_actions, action_obs_pairs, extra_info = self.env.act(command)
        if description is not None:
            if self.is_move_command(command):
                self.another_room_info = self.updated_description
            description = self.update_description(description)
            self.updated_description = description
            self.set_to_body(description, inventory, available_actions, action_obs_pairs)
            sys, usr = prompt_from_env_feedback(self.description, self.inventory, self.available_actions, self.action_obs_pairs, self.another_room_info)
            self.current_sys = sys
            self.current_usr = usr
        else: # 执行失败
            self.won = extra_info['won']
            self.action_obs_pairs.append((command, extra_info['raw_obs']))
            sys, usr = prompt_from_env_feedback(self.description, self.inventory, self.available_actions, self.action_obs_pairs, self.another_room_info)
            self.current_sys = sys
            self.current_usr = usr
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

def auto_play(game_index, action_list):
    game = Game_interface(game_index)
    game.reset()
    for action in action_list:
        game.input(action)
    return game