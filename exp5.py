# 测试another room info和desc simplify的效果，为了省钱只实验3次
from env import Env

def test_gpt4o_desc_only_simplify(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4O
    from llm_simplify_baseline import GPT_caller_simple_desc_only
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_simple_desc_only(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_claude_desc_only_simplify(hard_test_index = 0, filename_prefix = ''):
    from claude_simplify_baseline import Claude_caller_simple_desc_only
    from claude_simplify import CLAUDE
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = Claude_caller_simple_desc_only(env, gpt_type = CLAUDE, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller


def run_batch(start = 0, end = 3):
    for i in range(start, end):
        for game_index in range(5):
            test_gpt4o_desc_only_simplify(game_index, f'B{i}')
