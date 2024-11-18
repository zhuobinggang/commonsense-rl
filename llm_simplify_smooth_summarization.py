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


## another room info

class GPT_caller_simplify_desc_only_smooth_another_room(GPT_caller_simplify_desc_only_smooth_summarization):
    def env_act_succeed_callback(self):
        # Before desc update
        command = self.current_command
        if command and command.startswith('go ') and hasattr(self, 'desc_after_update'):
            print('XXXXXXXXXXXXXXXXXX   ')
            self.another_room_info = self.desc_after_update
        else:
            if not hasattr(self, 'desc_after_update'):
                self.another_room_info = 'unknown'
            else: # 如果已经设置过another room info，不需要重置为unknown
                pass

    def build_prompt(self, desc, inventory, available_actions, act_obs_pairs):
        self.builder.build(desc,
                           inventory,
                           available_actions,
                           act_obs_pairs,
                           another_room_info=self.another_room_info)
    
    def file_name_generate(self):
        shot = 'ZERO_SHOT'
        augment = 'AUGMENT_ON'
        filename_prefix = self.filename_prefix + '_' if self.filename_prefix else ''
        self.filename_raw = f'{filename_prefix}ANOTHER_ROOM_DESC_SIMPLE_SMOOTH_{shot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        self.filename = self.filename_raw + '.pkl'
        print(self.filename_raw)