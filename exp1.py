## EXP with simple action list + simple description. Started at 2024.8.12.

from env import Env
from llm_caller import Builder1, GPT_Caller
# from small_llm_summarizer import GPT_Caller_Simple_Desc
# from simple_option_exp import Builder_Simple_Option


# 这个就用之前的结果即可
def test_baseline_gpt4o(hard_test_index = 0, filename_prefix = ''):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-2024-08-06', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = Builder1(), filename_prefix = filename_prefix)
    caller.act_until_error(None)
    return caller

# - one-shot
def test_gpt4o_without_oneshot(hard_test_index = 0, filename_prefix = ''):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller(env, zero_shot = True, gpt_type = 'gpt-4o-2024-08-06', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = Builder1(), filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

# - one-shot
def test_gpt4o_without_cot(hard_test_index = 0, filename_prefix = ''):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-2024-08-06', cot = False, one_shot_easy = False, no_augment = False, step_limit = 20, builder = Builder1(), filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def run_batch(start = 0, end = 5):
    for i in range(start, end):
        for game_index in range(5):
            test_gpt4o_without_cot(game_index, f'B{i}')

# 这个结果也出来了，是描述和行动选项一起精简化的
def test_simple_desc_gpt4o(hard_test_index = 0):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller_Simple_Desc(env, zero_shot = False, gpt_type = 'gpt-4o-2024-08-06', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = Builder_Simple_Option())
    # caller = GPT_Caller_Simple_Desc(env, zero_shot = False, gpt_type = 'gpt-4o-2024-08-06', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = Builder1())
    caller.act_until_error(None)
    return caller
    
# 这个只测试行动选项精简化看看
def test_simple_option_gpt4o(hard_test_index = 0):
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-2024-08-06', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = Builder_Simple_Option())
    caller.act_until_error(None)
    return caller
