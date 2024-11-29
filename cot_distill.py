# chain of thought蒸馏
import json
from exp.finetune_4omini.reader import read_lines_by_file, command_from_line
from human_play import Game_interface
import global_variable as G
import common
from llm_simplify import GPT4OMINI, get_client

example_data_path = "exp/finetune_4omini/finetune_dataset_0_level_2_game_0.json"

def commands_by_game(game_index = 0):
    game_path = f"exp/finetune_4omini/finetune_dataset_0_level_2_game_{game_index}.json"
    lines = read_lines_by_file(game_path)
    commands = [command_from_line(line) for line in lines]
    return commands


def data_paths():
    results = []
    for i in range(5):
        results.append(f"exp/finetune_4omini/finetune_dataset_0_level_2_game_{i}.json")
    return results

def test():
    print(command_from_line(read_lines_by_file(example_data_path)[0]))

def game_played(game_index = 0):
    game = Game_cot_distill(game_index)
    commands = commands_by_game(game_index)
    game.auto_play(commands)
    return game

def save_pkl_with_game_index(game_index, pkl):
    import pickle
    with open(f'exp/auto_filename/game_{game_index}_human_play_llm_reason.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(pkl, outp, pickle.HIGHEST_PROTOCOL)

def game_played_then_request_reason_abstract(game_index, dic_to_sys_usr_func, quest_get_reason_func, save_pkl_func):
    game = game_played(game_index)
    prompt_necessary_info_dics = game.prompt_necessary_info_dics
    for dic in prompt_necessary_info_dics:
        sys, usr = dic_to_sys_usr_func(dic)
        response = quest_get_reason_func(sys, usr)
        dic['reason'] = response
    save_pkl_func(game_index, prompt_necessary_info_dics)




def game_played_then_request_reason_legacy(game_index = 0):
    game = game_played(game_index)
    prompt_necessary_info_dics = game.prompt_necessary_info_dics
    for dic in prompt_necessary_info_dics:
        sys, usr = dic_to_sys_usr(dic)
        response = quest_get_reason(sys, usr)
        dic['reason'] = response
    import pickle
    with open(f'exp/finetune_4omini_cot_distill/game_{game_index}_play_reason_4omini.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(prompt_necessary_info_dics, outp, pickle.HIGHEST_PROTOCOL)
    return prompt_necessary_info_dics

def game_played_then_request_reason(game_index = 0):
    game_played_then_request_reason_abstract(game_index, dic_to_sys_usr, quest_get_reason, save_pkl_with_game_index)

def load_reason_dics(file_path):
    import pickle
    f = open(file_path, 'rb')
    dics = pickle.load(f)
    f.close()
    return dics

class Prompt_builder_for_cot_distll:
    def __init__(self):
        self.task = G.TASK_COT_DISTLL
        self.action_history = ''
        self.inventory = ''
        self.another_room_info = ''
        self.current_enviroment = ''
        self.action_list = ''
        # NOTE
        self.best_action = ''
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
        user_msg += f'Next action: {self.best_action}\n' if self.best_action else ''
        user_msg += 'Reason (answer directly):'
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'


def sys_usr_cot_distill(description, inventory, available_actions, action_obs_pairs, another_room_info, best_action):
    promptor = Prompt_builder_for_cot_distll()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.action_list = common.actions_to_list(available_actions)
    promptor.another_room_info = another_room_info# TODO
    promptor.best_action = best_action
    promptor.build()
    return promptor.system_msg, promptor.user_msg

def dic_to_sys_usr(dic):
    if not 'command' in dic:
        return None, None
    return sys_usr_cot_distill(dic['description'], dic['inventory'], dic['available_actions'], dic['action_obs_pairs'], dic['another_room_info'], dic['command'])

class Game_cot_distill(Game_interface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_necessary_info_dics = []
        self.filename = f'finetuned_cot_distill_{self.dataset_index}_level_{self.hard_level_index}_game_{self.game_index}'
    def input(self, command, consideration = ''):
        self.command = command
        dic = {'description': self.description, 
               'inventory': self.inventory, 
               'available_actions': self.available_actions.copy(), 
               'action_obs_pairs': self.action_obs_pairs.copy(),
               'another_room_info': self.another_room_info,
               'command': self.command,
               'consideration': consideration}
        self.prompt_necessary_info_dics.append(dic)
        self.act_and_output(command)
    def save_as_json(self):
        f = open(f'exp/auto_filename/{self.filename}.jsonl', 'w')
        for dic in self.prompt_necessary_info_dics:
            sys, usr = sys_usr_cot_distill_training(dic['description'], dic['inventory'], dic['available_actions'], dic['action_obs_pairs'], dic['another_room_info'])
            obj = {'messages': [{"role": "system", "content": sys.strip()}, {"role": "user", "content": usr.strip()}, {"role": "assistant", "content": dic['command'].strip()}]}
            f.write(str(obj) + '\n')
        f.close()
    def save_readable(self):
        f = open(f'exp/auto_filename/{self.filename}.txt', 'w')
        for dic in self.prompt_necessary_info_dics:
            sys, usr = sys_usr_cot_distill_training(dic['description'], dic['inventory'], dic['available_actions'], dic['action_obs_pairs'], dic['another_room_info'])
            f.write(sys + usr + '\n' + dic['consideration'] + '\n' + dic['command'] + '\n\n')
        f.close()


def quest_get_reason(system_msg,
                        user_msg,
                        gpt_type=GPT4OMINI,
                        verbose = True):
    client = get_client()
    completion = client.chat.completions.create(
        model=gpt_type,  # 
        messages=[{
            "role": "system",
            "content": system_msg
        }, {
            "role": "user",
            "content": user_msg
        }],
        temperature=0)
    text = completion.choices[0].message.content
    if verbose:
        print(text)
    usage = str(completion.usage)
    return text


class Prompt_builder_for_cot_distll_training:
    def __init__(self):
        self.task = G.TASK_FINETUNE
        self.action_history = ''
        self.inventory = ''
        self.another_room_info = ''
        self.current_enviroment = ''
        self.action_list = ''
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
        user_msg += 'Answer (with consideration and the action, in json form):'
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'


def sys_usr_cot_distill_training(description, inventory, available_actions, action_obs_pairs, another_room_info):
    promptor = Prompt_builder_for_cot_distll_training()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.action_list = common.actions_to_list(available_actions)
    promptor.another_room_info = another_room_info# TODO
    promptor.build()
    return promptor.system_msg, promptor.user_msg

def dics_random_train_file_prepare(directory_path, out_path = 'exp/finetune_4omini_cot_distill/cot_distill_training.jsonl'):
    import os
    # 遍历文件夹中的所有文件
    dics = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pkl'):  # 只处理 JSON 文件
            file_path = os.path.join(directory_path, file_name)
            temp_dics = load_reason_dics(file_path)
            dics += temp_dics
    import random
    random.shuffle(dics)
    # out.jsonl prepare
    lines = []
    import json
    for dic in dics:
        sys, usr = sys_usr_cot_distill_training(dic['description'], dic['inventory'], dic['available_actions'], dic['action_obs_pairs'], dic['another_room_info'])
        assistant = json.dumps({'consideration': dic['reason'].replace('\n',''), 'action': dic['command']})
        obj = {'messages': [{"role": "system", "content": sys.strip()}, {"role": "user", "content": usr.strip()}, {"role": "assistant", "content": assistant.strip()}]}
        line = json.dumps(obj)
        lines.append(line)
    f = open(out_path, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()

# upload & finetune

training_data_path = 'exp/finetune_4omini_cot_distill/cot_distill_training.jsonl'

