from human_play import *
from llm_caller import get_client
import global_variable as G

def quest_get_command(system_msg,
                        user_msg,
                        # gpt_type=GPT4O_FINETUNE_1EPOCH,
                        gpt_type = G.GPT4o_mini,
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
    command = completion.choices[0].message.content
    if verbose:
        print(command)
    usage = str(completion.usage)
    return command

# @legacy: only for twc
def llm_auto_play(game_index, file_prefix = 'B0_'):
    game = Game_interface(game_index, dataset_index=2, hard_level_index=2) # test valid set
    game.reset()
    print(f'MAX SCORE: {game.env.env.info["max_score"]}')
    count = 0
    max_try = 20
    while count < max_try and not game.won:
        count += 1
        try:
            sys = game.current_sys
            usr = game.current_usr
            command = quest_get_command(sys, usr)
            game.input(command)
        except:
            print('EXCEPT')
            break
    game.filename = f'{file_prefix}finetuned_play_game{game_index}_score{game.env.env.last_reward}.jsonl'
    game.save_as_json()
    game.save_readable()
    return game

def batch(start = 0, end = 5):
    for batch_index in range(start, end):
        for game_index in range(5):
            _ = llm_auto_play(game_index, f'B{batch_index}_')


def fake_sys_usr_from_game_func(game):
    return 'sys', 'usr'

def llm_auto_play_action_selector(game, game_index = 999, testing = True, file_prefix = 'B0_', max_try = 20, gpt_type = G.GPT4o_mini, sys_usr_from_game_func = fake_sys_usr_from_game_func):
    from action_selector import quest_closet_action
    print(gpt_type)
    game.reset()
    count = 0
    log_triples = []
    while count < max_try and not game.won and not game.lost:
        count += 1
        try:
            sys, usr = sys_usr_from_game_func(game)
            command = quest_get_command(sys, usr, gpt_type=gpt_type).strip()
            log_triples.append((sys, usr, command))
            available_actions = game.available_actions + ['inventory']
            if command not in available_actions: # DONE: 调用action selctor
                org_command = command
                command = quest_closet_action(game.available_actions, org_command).strip()
                print(f'调用action selector: {org_command}->{command}')
                log_triples.append(('ACTION SELCTOR', 'ACTION SELCTOR', f'{org_command}->{command}'))
            _ = game.input(command)
        except:
            print('EXCEPT')
            break
    if testing:
        f = open(f'exp/auto_filename/testing_{game_index}_{gpt_type}_{file_prefix}finetuned_play_game_score{game.env.env.last_reward}_of_{game.env.env.max_score}.txt', 'w')
    else:
        f = open(f'exp/auto_filename/valid_{game_index}_{gpt_type}_{file_prefix}finetuned_play_game_score{game.env.env.last_reward}_of_{game.env.env.max_score}.txt', 'w')
    for sys, usr, command in log_triples:
        f.write(sys + '\n' + usr + '\n' + command + '\n\n')
    print(f'##################  GAME {game_index} WROTE!!!')
    f.close()
    return game


def llm_auto_play(game, game_index, testing = True, file_prefix = 'B0_', max_try = 20, gpt_type = G.GPT4o_mini, sys_usr_from_game_func = fake_sys_usr_from_game_func):
    from finetuned_play import quest_get_command
    # from action_selector import quest_closet_action
    print(gpt_type)
    game.reset()
    count = 0
    log_triples = []
    while count < max_try and not game.won and not game.lost:
        count += 1
        try:
            sys, usr = sys_usr_from_game_func(game)
            command = quest_get_command(sys, usr, gpt_type=gpt_type).strip()
            log_triples.append((sys, usr, command))
            _ = game.input(command)
        except:
            print('EXCEPT')
            break
    if testing:
        f = open(f'exp/auto_filename/testing_{game_index}_{gpt_type}_{file_prefix}finetuned_play_game_score{game.env.env.last_reward}_of_{game.env.env.max_score}.txt', 'w')
    else:
        f = open(f'exp/auto_filename/valid_{game_index}_{gpt_type}_{file_prefix}finetuned_play_game_score{game.env.env.last_reward}_of_{game.env.env.max_score}.txt', 'w')
    for sys, usr, command in log_triples:
        f.write(sys + '\n' + usr + '\n' + command + '\n\n')
    print(f'##################  GAME {game_index} WROTE!!!')
    f.close()
    return game