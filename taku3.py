from env import Env
from llm_caller import GPT_Caller, Builder1
from simple_option_exp import Builder_Simple_Option

def test(hard_test_index = 0):
    env = Env(2, hard_test_index, 1, no_augment = False)
    builder = Builder_Simple_Option()
    caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-mini-2024-07-18', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = builder)
    # caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4-1106-preview', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = builder)
    caller.act_and_call(None)
    return caller


# def test_baseline_gpt4o(hard_test_index = 0):
#     env = Env(2, hard_test_index, 1, no_augment = True)
#     builder = Builder1()
#     # caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-mini-2024-07-18', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = builder)
#     caller = GPT_Caller(env, zero_shot = True, gpt_type = 'gpt-4-1106-preview)', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = builder)
#     caller.act_and_call(None)
#     return caller
# 
