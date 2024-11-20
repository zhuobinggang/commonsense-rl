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
