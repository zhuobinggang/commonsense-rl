from human_play import *
from llm_caller import get_client
GPT4O_FINETUNE_1EPOCH = 'ft:gpt-4o-mini-2024-07-18:personal::AXLvStgP'

def quest_get_command(system_msg,
                        user_msg,
                        gpt_type=GPT4O_FINETUNE_1EPOCH,
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

def llm_auto_play(game_index, file_prefix = 'B0_'):
    game = Game_interface(game_index, 1, 2)
    game.reset()
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

def batch(start, end):
    for batch_index in range(start, end):
        for game_index in range(5):
            _ = llm_auto_play(game_index, f'B{batch_index}_')
