# 2024.11.18 不使用完全形式化的房间信息简化
from llm_simplify import *

class Summarization_prompt_builder_smooth:

    def __init__(self):
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''
        self.desc = ''

    def build(self):
        system_msg = ''
        system_msg += 'Task: Simplify the environment description.\n'
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += 'The environment description:\n'
        user_msg += self.desc.strip().replace('\n', '')
        user_msg = user_msg.strip() + '\n'
        user_msg += 'Response (Response with the new environment description directly):\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

    def sys_usr_msg(self):
        return self.system_msg, self.user_msg
    

class GPT_caller_simplify_smooth_summarization(GPT_Caller_Simplify):
    def init_summarize_prompt_builder(self):
        self.summarize_prompt_builder = Summarization_prompt_builder_smooth() # full action list
        

class GPT_caller_simplify_desc_only_smooth_summarization(GPT_caller_simplify_smooth_summarization):
    def init_prompt_builder(self, builder):
        self.builder = builder or Builder_old_style()