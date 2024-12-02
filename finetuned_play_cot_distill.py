from finetuned_play import *
from llm_simplify import text_from_raw_response
from common import json_obj_from_text
from cot_distill import Game_cot_distill, sys_usr_cot_distill_training

E1 = 'ft:gpt-4o-mini-2024-07-18:personal::AY6Y2SEQ'
E2 = 'ft:gpt-4o-mini-2024-07-18:personal::AYR46lsF'
E3 = 'ft:gpt-4o-mini-2024-07-18:personal::AYSdQn8v'
MODELS = [E1, E2, E3]

def quest_get_consideration_and_command(system_msg,
                        user_msg,
                        gpt_type=E3,
                        verbose = False):
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
    # To clipboard
    usage = str(completion.usage)
    # To clipboard
    # NOTE: 自动提取command
    text = text_from_raw_response(completion)
    obj = json_obj_from_text(text)
    if verbose:
        print(text)
        print('------------------>')
        print(obj)
    dic = {'response': obj, 'usage': usage}
    the_command = obj['action'] # Might be None
    consideration = obj['consideration']
    return consideration, the_command

def llm_auto_play(game_index, file_prefix = 'B0_', max_try = 20, gpt_type = E3):
    print(gpt_type)
    game = Game_cot_distill(game_index, 1, 2)
    game.reset()
    count = 0
    while count < max_try and not game.won:
        count += 1
        try:
            sys, usr = sys_usr_cot_distill_training(game.description, game.inventory, game.available_actions, game.action_obs_pairs, game.another_room_info)
            consideration, command = quest_get_consideration_and_command(sys, usr, gpt_type=gpt_type)
            game.input(command, consideration)
        except:
            print('EXCEPT')
            break
    game.filename = f'{file_prefix}finetuned_play_game{game_index}_score{game.env.env.last_reward}.jsonl'
    game.save_as_json()
    game.save_readable()
    return game

def llm_auto_play_valid_set(game_index, file_prefix = 'B0_', max_try = 20, gpt_type = MODELS[1]):
    print(f'VALID gpt_type: {gpt_type}')
    game = Game_cot_distill(game_index, 2, 2) # game_index, dataset_index, hard_level
    game.reset()
    count = 0
    while count < max_try and not game.won:
        count += 1
        try:
            sys, usr = sys_usr_cot_distill_training(game.description, game.inventory, game.available_actions, game.action_obs_pairs, game.another_room_info)
            consideration, command = quest_get_consideration_and_command(sys, usr, gpt_type=gpt_type)
            game.input(command, consideration)
        except:
            print('EXCEPT')
            break
    game.filename = f'valid_{gpt_type}_{file_prefix}finetuned_play_game{game_index}_score{game.env.env.last_reward}.jsonl'
    game.save_as_json()
    game.save_readable()
    return game

def batch(start = 0, end = 1, gpt_type = E3):
    for batch_index in range(start, end):
        for game_index in range(5):
            _ = llm_auto_play(game_index, f'B{batch_index}_', gpt_type=gpt_type)


def batch_valid(start = 0, end = 1):
    for batch_index in range(start, end):
        for e in range(3):
            for game_index in range(5):
                _ = llm_auto_play_valid_set(game_index, f'B{batch_index}_', gpt_type=MODELS[e])