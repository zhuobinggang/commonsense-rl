# 微调4omini，简化desc&行动列表
from cot_distill import commands_by_game, Game_cot_distill
import common
from llm_simplify import quest_4omini_simplify_desc
from human_play import Prompt_builder_for_finetune

# ========= 准备每个游戏对应的training file，剩下要做的事情只有全部读取并打乱 =======================

def prompt_from_env_feedback_simple_desc(description, inventory, available_actions, action_obs_pairs, another_room_info):
    promptor = Prompt_builder_for_finetune()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    prompt, simplified_current_env = quest_4omini_simplify_desc(description)
    print(simplified_current_env)
    promptor.current_enviroment = simplified_current_env # NOTE
    promptor.action_list = common.actions_to_list(available_actions)
    if len(another_room_info) > 10:
        prompt, simplified_another_room_info = quest_4omini_simplify_desc(another_room_info) # NOTE
        print('Another room info simplify: ')
        print(simplified_another_room_info)
    else: # 不存在另一个房间信息的情况
        simplified_another_room_info = another_room_info
    promptor.another_room_info = simplified_another_room_info
    promptor.build()
    return promptor.system_msg, promptor.user_msg

def dic_to_sys_usr_func_simple_desc(dic):
    return prompt_from_env_feedback_simple_desc(dic['description'], dic['inventory'], dic['available_actions'], dic['action_obs_pairs'], dic['another_room_info'])

class Game_simplify(Game_cot_distill):
    def save_as_json(self):
        print('SAVE IT IN LLM PLAY FUNCTION YOURSELF!!!!!')
    def save_readable(self):
        print('SAVE IT IN LLM PLAY FUNCTION YOURSELF!!!!!')
    def available_actions_got_callback(self, available_actions):
        # NOTE: save movable direction
        self.movable_direction = ''
        if not available_actions:
            return
        for action in available_actions:
            if action.startswith('go'):
                self.movable_direction = action.replace('go ', '')
                print('movable direction updated: ' + self.movable_direction)
    def input(self, command, consideration = ''):
        self.command = command
        dic = {'description': self.description, 
               'inventory': self.inventory, 
               'available_actions': self.available_actions.copy(), 
               'action_obs_pairs': self.action_obs_pairs.copy(),
               'another_room_info': self.another_room_info,
               'command': self.command,
               'consideration': consideration,
               'movable_direction': self.movable_direction} # NOTE
        self.prompt_necessary_info_dics.append(dic)
        self.act_and_output(command)

def game_played(game_index = 0):
    game = Game_simplify(game_index)
    commands = commands_by_game(game_index)
    game.auto_play(commands)
    return game

def game_played_and_save_training_file(game_index, output_file_path = 'exp/auto_filename/dd.jsonl'):
    import json
    game = game_played(game_index)
    prompt_necessary_info_dics = game.prompt_necessary_info_dics
    f = open(output_file_path, 'w')
    for dic in prompt_necessary_info_dics:
        sys, usr = dic_to_sys_usr_func_simple_desc(dic) # NOTE: 发送简化请求
        agent = dic['command']
        obj = common.training_line_prepare(sys, usr, agent)
        line = json.dumps(obj)
        f.write(line + '\n')
    f.close()


def test():
    game_index = 0
    game_played_and_save_training_file(game_index, output_file_path=f'exp/auto_filename/train_simplify_game{game_index}.jsonl')

def batch_train_file_prepare():
    for game_index in range(5):
        game_played_and_save_training_file(game_index, output_file_path=f'exp/auto_filename/train_simplify_game{game_index}.jsonl')

# ================================= END ========================

# ================================ 读取并打乱 ========================

def lines_random_train_file_prepare(directory_path = 'exp/auto_filename', out_path='exp/auto_filename/training.jsonl'):
    """
    Reads all .jsonl files in the given directory, shuffles the lines, and writes them to an output file.

    Args:
        directory_path (str): The directory containing JSONL files.
        out_path (str): The output path for the shuffled training file.
    """
    import os
    import json
    import random
    all_lines = []

    # Traverse the directory to find all JSONL files
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                
                # Read lines from the current JSONL file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Ensure the line is valid JSON
                        try:
                            json.loads(line)
                            all_lines.append(line)
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line in file {file_path}: {line.strip()}")

    # Shuffle all the lines
    random.shuffle(all_lines)

    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write shuffled lines to the output file
    with open(out_path, 'w', encoding='utf-8') as out_file:
        out_file.writelines(all_lines)

    print(f"Shuffled lines written to {out_path}")

# ====================== END =======================

# ====================== 检验文件 & 上传 & 微调 ========================

# ### validation
training_data_path ='exp/finetune_4omini_simplify_desc/training.jsonl'

def valid(training_data_path = training_data_path):
    from exp.finetune_4omini.varidation import load_ds, format_valid, ds_info, cost_est
    data_path = training_data_path
    ds = load_ds(data_path)
    format_valid(ds)
    dd = ds_info(ds)
    cost_est(ds, dd)

# Result: file-4ABmkrsyPrVYxQ5r2vyBLq
def finetune_file_upload(training_data_path = training_data_path):
    from exp.finetune_4omini.finetune import create_file
    return create_file(training_data_path)

UPLOADED_FILE_ID = 'file-5L2WHriDUkQqJ9RRXZxYcZ'

def finetune(UPLOADED_FILE_ID = UPLOADED_FILE_ID):
    from exp.finetune_4omini.finetune import fine_tune
    return fine_tune(UPLOADED_FILE_ID)

E1 = 'ft:gpt-4o-mini-2024-07-18:personal::AZtikUfb:ckpt-step-70'
E2 = 'ft:gpt-4o-mini-2024-07-18:personal::AZtikEzM:ckpt-step-140'
E3 = 'ft:gpt-4o-mini-2024-07-18:personal::AZtikr70'
MODELS = [E1, E2, E3]

# ==================== 重新实现llm play，需要实时将desc简化 ==================
def sys_usr_from_game(game):
    return prompt_from_env_feedback_simple_desc(game.description, game.inventory, game.available_actions, game.action_obs_pairs, game.another_room_info)

def llm_auto_play(game_index, testing = True, file_prefix = 'B0_', max_try = 20, gpt_type = E3, sys_usr_from_game_func = sys_usr_from_game):
    from finetuned_play import quest_get_command
    print(gpt_type)
    if testing:
        game = Game_simplify(game_index, 1, 2) # test set
    else:
        game = Game_simplify(game_index, 2, 2) # valid set
    game.reset()
    count = 0
    log_triples = []
    while count < max_try and not game.won:
        count += 1
        try:
            sys, usr = sys_usr_from_game_func(game)
            command = quest_get_command(sys, usr, gpt_type=gpt_type)
            game.input(command)
            log_triples.append((sys, usr, command))
        except:
            print('EXCEPT')
            break
    if testing:
        f = open(f'exp/auto_filename/testing_{gpt_type}_{file_prefix}finetuned_play_game{game_index}_score{game.env.env.last_reward}.txt', 'w')
    else:
        f = open(f'exp/auto_filename/valid_{gpt_type}_{file_prefix}finetuned_play_game{game_index}_score{game.env.env.last_reward}.txt', 'w')
    for sys, usr, command in log_triples:
        f.write(sys + '\n' + usr + '\n' + command + '\n\n')
    f.close()
    return game

def valid_play():
    for e in range(3):
        for game_index in range(5):
            _ = llm_auto_play(game_index, testing=False, file_prefix='B0_', gpt_type=MODELS[e])

def test_play(gpt_type):
    for game_index in range(5):
        _ = llm_auto_play(game_index, testing=True, file_prefix='B0_', gpt_type=gpt_type)