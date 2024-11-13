from env import Env
from llm_simplify import GPT_Caller_Simplify, GPT4O
from llm_simplify_baseline import GPT_Caller_Baseline

def test_gpt4o_baseline(hard_test_index = 0, filename_prefix = ''):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller_Baseline(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_gpt4o_simplified(hard_test_index = 0, filename_prefix = ''):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller_Simplify(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def run_batch(start = 0, end = 5):
    for i in range(start, end):
        for game_index in range(5):
            test_gpt4o_simplified(game_index, f'B{i}')

def test():
    hard_test_index = 0
    filename_prefix = 'test'
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller_Simplify(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    return caller