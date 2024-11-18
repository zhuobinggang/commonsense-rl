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

    def __call__(self, description, inventory, available_actions,
                 action_obs_pairs, need_print = False):
        self.builder.build(description,
                           inventory,
                           available_actions,
                           action_obs_pairs,
                           another_room_info=self.another_room_info)
        system_msg, user_msg = self.builder.sys_usr_msg()
        complete, dic, the_command = self.quest_my_llm(system_msg,
                                             user_msg,
                                             self.gpt_type) # 获得the_command，可能为空
        if the_command is None:
            print('__call__(): QUEST LLM GET NONE COMMAND, THE RESPONSE DIC IS BELOW, I WILL TRY IT AGAIN!')
            print(dic)
        if self.env is not None:
            self.env.env.system_user_msgs.append(system_msg + user_msg) # 2024.11.9: 不管command是否为空，都存储sys, user信息，这个可以用于再次请求才对。
            self.env.env.gpt_responses.append(complete)
            self.env.env.readable_log += (system_msg + user_msg + '\n\n\n' +
                                          str(dic['response']) + '\n\n\n\n')
        return the_command
    
    def file_name_generate(self):
        shot = 'ZERO_SHOT'
        augment = 'AUGMENT_ON'
        filename_prefix = self.filename_prefix + '_' if self.filename_prefix else ''
        self.filename_raw = f'{filename_prefix}ANOTHER_ROOM_DESC_SIMPLE_{shot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        self.filename = self.filename_raw + '.pkl'
        print(self.filename_raw)