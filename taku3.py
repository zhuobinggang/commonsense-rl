from env import Env
from llm_caller import GPT_Caller

def test(hard_test_index = 0):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-mini-2024-07-18', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20)
    caller.act_and_call(None)
    return caller
