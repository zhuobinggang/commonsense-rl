# 2024.11.18 测试更加温和的环境summarization
from env import Env

def test_gpt4o_desc_only_simplify_smooth(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4O
    from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_summarization
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_simplify_desc_only_smooth_summarization(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_4omini_desc_only_simplify_smooth(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4OMINI
    from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_summarization
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_simplify_desc_only_smooth_summarization(env, gpt_type = GPT4OMINI, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

# another room info
def test_4omini_desc_only_simplify_smooth_another_room(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4OMINI
    from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_another_room
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_simplify_desc_only_smooth_another_room(env, gpt_type = GPT4OMINI, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_gpt4o_desc_only_simplify_smooth_another_room(hard_test_index = 0, filename_prefix = ''):
    from llm_simplify import GPT4O
    from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_another_room
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = GPT_caller_simplify_desc_only_smooth_another_room(env, gpt_type = GPT4O, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

# claude

def test_claude_desc_smooth_another_room(hard_test_index = 0, filename_prefix = ''):
    from claude_simplify_smooth_summarization import Claude_caller_smooth_desc_another_room
    from claude_caller import CLAUDE_TYPE
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = Claude_caller_smooth_desc_another_room(env, gpt_type = CLAUDE_TYPE, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def test_claude_desc_smooth(hard_test_index = 0, filename_prefix = ''):
    from claude_simplify_smooth_summarization import Claude_caller_smooth_desc
    from claude_caller import CLAUDE_TYPE
    env = Env(2, hard_test_index, 1, no_augment = False)
    caller = Claude_caller_smooth_desc(env, gpt_type = CLAUDE_TYPE, step_limit = 20, filename_prefix=filename_prefix)
    caller.act_until_error(None)
    return caller

def run_batch(start = 0, end = 3):
    for i in range(start, end):
        for game_index in range(5):
            test_claude_desc_smooth_another_room(game_index, f'B{i}')