from env import Env

def test_gpt4o_baseline(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT_Caller_Simplify, GPT4O
    from llm_simplify_baseline import GPT_Caller_Baseline
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller_Baseline(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_gpt4o_simplified(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT_Caller_Simplify, GPT4O
    from llm_simplify_baseline import GPT_Caller_Baseline
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_Caller_Simplify(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

### claude


def test_claude_baseline(hard_test_index = 0, filename_prefix = ''):
    from claude_simplify_baseline import Claude_caller_baseline
    from claude_simplify import CLAUDE
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = Claude_caller_baseline(env, gpt_type = CLAUDE, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_claude_simplified(hard_test_index = 0, filename_prefix = ''):
    from claude_simplify import Claude_caller_simplify, CLAUDE
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = Claude_caller_simplify(env, gpt_type = CLAUDE, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller


def run_batch(start = 0, end = 5):
    for i in range(start, end):
        for game_index in range(5):
            test_claude_baseline(game_index, f'B{i}')