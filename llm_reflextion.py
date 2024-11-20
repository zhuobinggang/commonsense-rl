from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_another_room
from llm_simplify import Builder_old_style, GPT4OMINI, quest_simple_get_text

# TODO: 完成prompt的建构
class Summarization_Prompt_builder:
    def __init__(self):
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''
        self.desc = ''

    def build(self):
        system_msg = ''
        system_msg += '\n'
        system_msg += 'Example: Bedroom[wardrobe, chest of drawers[black sock], desk[pen, eraser]]\n'
        system_msg += 'Response with the new environment description directly.\n'
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += 'The environment description:\n'
        user_msg += self.desc.strip().replace('\n', '')
        user_msg = user_msg.strip() + '\n'
        user_msg += 'Response:\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

    def sys_usr_msg(self):
        return self.system_msg, self.user_msg
    
class Builder_full_action_list_with_reflextion(Builder_old_style):
    def action_obs_pairs_to_history(self, action_obs_pairs):
        action_history = ''
        action_idx = 0
        if len(action_obs_pairs) > 0:
            for _, (act, obs) in enumerate(action_obs_pairs):
                if act.startswith('Refle'):
                    action_history += f'Reflection: {obs} ';
                else:
                    action_history += f'Action {action_idx}: {act} -> {obs} '
                    action_idx += 1
        else:
            action_history = 'No action was taken now.'
        return action_history

def quest_reflextion(system_msg,
                        user_msg,
                        gpt_type=GPT4OMINI,
                        verbose = True):
    return quest_simple_get_text(system_msg, user_msg, gpt_type, verbose)


# 2024.11.20 Reflextion
class GPT_agent_smooth_desc_another_room_reflextion(GPT_caller_simplify_desc_only_smooth_another_room):
    def try_adjust_and_execute(self, command): # 2024.11.20
        available_actions = self.env.available_actions;
        if command not in available_actions:
            return None, None, None, None, {'raw_obs': 'The command is not in the available command list, please recheck.', 'command_not_in_list': True, 'failed': True};
        else:
            description, inventory, available_actions, action_obs_pairs, extra_info = self.env.act(command, no_augment=self.no_augment)
            return description, inventory, available_actions, action_obs_pairs, extra_info
        
    def call_llm_get_next_command_try_twice(self):
        extra_info = {}
        command_may_none = self.recall_and_get_command()
        if command_may_none is None:
            self.add_to_readable_log('\n[LLM ERR] The response from LLM was structured incorrectly, Need request again.\n')
            print('LLM的回复存在结构异常，将进行再次请求。')
            command_may_none = self.recall_and_get_command()
        if command_may_none is None:
            self.add_to_readable_log('\n[LLM ERR] The second response from LLM was structured incorrectly. No more try.\n')
            print('LLM的回复存在结构异常，不再重试。')
            extra_info['llm_response_failed'] = True;
        return  command_may_none, extra_info# 得到的是提取成功的，如果提取不成功，需要手动操作
    
    def update_observation(self, description, inventory, available_actions, action_obs_pairs):
        # NOTE: 由于步数增加意味着环境发生改变，需要更新环境描述
        self.desc_before_update = description
        description = self.updated_description(description)
        self.desc_after_update = description
        self.set_act_result_to_body(self.desc_after_update, inventory, available_actions, action_obs_pairs)
        self.append_score_by_step()

    def llm_reflextion(self, raw_obs):
        # TODO: 完成prompt的建构
        return 'I have no idea.'
    
    def add_reflextion_into_action_history(self, reflextion):
        # TODO: 完成
        pass

    # NOTE: 只考虑提供options的情况，所以可以忽视执行错误的情况
    def act_and_call(
            self,
            command=None):  # @RETURN: None means 2 path, first means the command non-executable, second means response from LLM irregular.
        extra_info = {}
        extra_info['llm_response_failed'] = False;
        extra_info['reach_step_limit'] = False;
        extra_info['won'] = False;
        extra_info['reflexted'] = False;
        if self.step_counter <= self.step_limit:
            self.current_command = command
            description, inventory, available_actions, action_obs_pairs, extra_info_raw = self.try_adjust_and_execute(command)
            extra_info = extra_info | extra_info_raw
            if extra_info['env_act_failed']: # 指令执行失败的情况，需要让模型反思，随后调用模型获取下一个指令，NOTE: 该情况下步数不会增加
                # NOTE: Reflextion: 让模型反思该错误并把结果放进action history
                self.env.append_command_obs_pair(command, extra_info['raw_obs'])
                reflextion = self.llm_reflextion(extra_info['raw_obs'])
                self.add_reflextion_into_action_history(reflextion)
                extra_info['reflexted'] = True;
                command_may_none, raw_extra_info = self.call_llm_get_next_command_try_twice()
                extra_info = extra_info | raw_extra_info
                return command_may_none, extra_info
            else: # NOTE: 指令执行成功，步数增加
                self.step_counter += 1
                self.env_act_succeed_callback()
                if self.is_end(): # 如果赢了就不再需要调用LLM
                    print('YOU WIN! NO API CALL NEED.')
                    self.add_to_readable_log('YOU WIN! NO API CALL NEED.')
                    self.save()
                    extra_info['won'] = True
                    return None, extra_info
                else:
                    # NOTE: 由于步数增加意味着环境发生改变，需要更新环境描述
                    self.update_observation(description, inventory, available_actions, action_obs_pairs)
                    command_may_none, raw_extra_info = self.call_llm_get_next_command_try_twice()
                    extra_info = extra_info | raw_extra_info
                    return command_may_none, extra_info
        else:
            print(f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}')
            self.save()
            extra_info = extra_info | {'reach_step_limit': True, 'raw_obs': f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}'}
        
    def act_until_error(self, command=None):
        next_command, extra_info = self.act_and_call(command)
        while (not extra_info['reach_step_limit']) and (not extra_info['won']):
            next_command, extra_info = self.act_and_call(next_command)
            if next_command is None:
                print('XXXXXXXXXXXXX 通过LLM获取下一个指令错误，请检查extra_info:')
                print(extra_info)
                return self
