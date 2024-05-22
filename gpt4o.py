from gpt4 import GPT_Caller, get_game_env, save_record
from taku_step_by_step import act_step_by_step

def run():
    env = get_game_env(2, 0)
    caller = GPT_Caller(zero_shot = True, gpt_type = 'gpt-4o-2024-05-13', env = env)
    dd = act_step_by_step(env, None, caller) # First action retrieve
