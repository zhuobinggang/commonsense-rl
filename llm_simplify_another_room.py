from llm_simplify import GPT_Caller_Simplify, Builder_old_style
from llm_simplify_baseline import GPT_caller_simple_desc_only


class GPT_caller_desc_simple_another_room(GPT_caller_simple_desc_only):

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
        self.filename_raw = f'{filename_prefix}ANOTHER_ROOM_DESC_SIMPLE_{shot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        self.filename = self.filename_raw + '.pkl'
        print(self.filename_raw)