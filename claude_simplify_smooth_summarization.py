## claude
from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_another_room, GPT_caller_simplify_desc_only_smooth_summarization
from claude_simplify import quest_claude_simple
from claude_caller import get_client

class Claude_caller_smooth_desc_another_room(GPT_caller_simplify_desc_only_smooth_another_room):
    def quest_my_llm(self, system_msg, user_msg, llm_type, need_print = False):
        complete, dic, the_command = quest_claude_simple(get_client(), system_msg,
                                        user_msg,
                                        claude_type=llm_type, verbose=need_print) # 获得the_command，可能为空
        return complete, dic, the_command
    

# 作为新的baseline
class Claude_caller_smooth_desc(GPT_caller_simplify_desc_only_smooth_summarization):
    def quest_my_llm(self, system_msg, user_msg, llm_type, need_print = False):
        complete, dic, the_command = quest_claude_simple(get_client(), system_msg,
                                        user_msg,
                                        claude_type=llm_type, verbose=need_print) # 获得the_command，可能为空
        return complete, dic, the_command