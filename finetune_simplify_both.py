from finetune_simplify_actions import game_played, dic_to_sys_usr_func_simple_action_list, Promptor_simple_actions
from llm_simplify import quest_4omini_simplify_desc
import common

# =============== train file prepare ================

def prompt_from_env_feedback_simple_desc_and_actions(description, inventory, action_obs_pairs, another_room_info, movable_direction):
    promptor = Promptor_simple_actions()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    _, simplified_current_env = quest_4omini_simplify_desc(description)
    promptor.current_enviroment = simplified_current_env # NOTE
    if len(another_room_info) > 10:
        _, simplified_another_room_info = quest_4omini_simplify_desc(another_room_info) # NOTE
    else: # 不存在另一个房间信息的情况
        simplified_another_room_info = another_room_info
    promptor.another_room_info = simplified_another_room_info
    promptor.movable_direction = movable_direction
    promptor.build()
    return promptor.system_msg, promptor.user_msg


def dic_to_sys_usr_func_simple_action_list(dic):
    return prompt_from_env_feedback_simple_desc_and_actions(dic['description'], dic['inventory'], dic['action_obs_pairs'], dic['another_room_info'], dic['movable_direction'])

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
        game_played_and_save_training_file(game_index, output_file_path=f'exp/auto_filename/train_simplify_both_game{game_index}.jsonl')

# ================== randomize training file & output ==================

def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare()

# ================== validation =================

training_data_path ='exp/finetune_4omini_simplify_both/training.jsonl'

def valid_then_upload():
    from finetune_simplify_desc import valid, finetune_file_upload
    valid(training_data_path)
    finetune_file_upload(training_data_path)

def finetune_both():
    FILE_ID = 'file-Ce2XsXS3Q7ogRe1NBmbeVz'
    from finetune_simplify_desc import finetune
    finetune(FILE_ID)


MODELS = ['ft:gpt-4o-mini-2024-07-18:personal::AadYwWBS:ckpt-step-70', 'ft:gpt-4o-mini-2024-07-18:personal::AadYxmyg:ckpt-step-140', 'ft:gpt-4o-mini-2024-07-18:personal::AadYxS0e']
BEST = MODELS[2]

# ================ play ==================

def sys_usr_from_game(game):
    return prompt_from_env_feedback_simple_desc_and_actions(game.description, game.inventory, game.action_obs_pairs, game.another_room_info, game.movable_direction)


def llm_auto_play(game_index, testing = True, file_prefix = 'B0_', max_try = 20, gpt_type = MODELS[2], sys_usr_from_game_func = sys_usr_from_game):
    from finetuned_play import quest_get_command
    from action_selector import quest_closet_action
    from finetune_simplify_desc import Game_simplify
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
            command = quest_get_command(sys, usr, gpt_type=gpt_type).strip()
            log_triples.append((sys, usr, command))
            if command not in game.available_actions:
                org_command = command
                command = quest_closet_action(game.available_actions, command).strip()
                print(f'调用action selector: {org_command}->{command}')
                log_triples.append(('ACTION SELCTOR', 'ACTION SELCTOR', f'{org_command}->{command}'))
            _ = game.input(command)
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

# DONE: 测试　
def test():
    llm_auto_play(0, testing=False, gpt_type = MODELS[0], max_try=10, sys_usr_from_game_func = sys_usr_from_game)

def batch_valid():
    for model in MODELS:
        for i in range(5):
            llm_auto_play(i, testing=False, gpt_type = model, max_try=20, sys_usr_from_game_func = sys_usr_from_game)


def batch_test(): # DONE: 0.79
    for i in range(5):
        llm_auto_play(i, testing=True, gpt_type = BEST, max_try=20, sys_usr_from_game_func = sys_usr_from_game)