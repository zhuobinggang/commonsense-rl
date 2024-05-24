from gpt4 import GPT_Caller, get_game_env, save_record

def run(game_idx = 0):
    env = get_game_env(2, game_idx)
    caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-2024-05-13', cot = True, one_shot_easy = False, no_augment = False)
    caller.step(None) # first step
    return caller

