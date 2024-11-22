# 测试Env_extra_info和llm_reflextion
from env import Env_extra_info

# another room info
def test_4omini_smooth_desc_another_room_reflextion(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4OMINI
    from llm_reflextion import GPT_agent_smooth_desc_another_room_reflextion
    env = Env_extra_info(2, hard_test_index, 1, no_augment = False)
    caller = GPT_agent_smooth_desc_another_room_reflextion(env, gpt_type = GPT4OMINI, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_4omini_consideration_buffer(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4OMINI
    from llm_consideration_buffer import GPT_caller_consideration_buffer
    env = Env_extra_info(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_consideration_buffer(env, gpt_type = GPT4OMINI, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller


def test_4omini_react_style(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4OMINI
    from llm_consideration_buffer import GPT_caller_react
    env = Env_extra_info(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_react(env, gpt_type = GPT4OMINI, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller


def run_batch(start = 0, end = 3):
    for i in range(start, end):
        for game_index in range(5):
            test_4omini_react_style(game_index, f'B{i}')