# another room info
from env import Env

def test_mini_simple_desc_another_room(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4OMINI
    from llm_simplify_another_room import GPT_caller_desc_simple_another_room
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_desc_simple_another_room(env, gpt_type = GPT4OMINI, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_4o_simple_desc_another_room(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4O
    from llm_simplify_another_room import GPT_caller_desc_simple_another_room
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_desc_simple_another_room(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def run_batch(start = 0, end = 5):
    for i in range(start, end):
        for game_index in range(5):
            test_mini_simple_desc_another_room(game_index, f'B{i}')