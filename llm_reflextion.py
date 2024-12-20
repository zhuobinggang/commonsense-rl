from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_another_room
from llm_simplify import Builder_old_style, GPT4OMINI, quest_simple_get_text
from common import actions_to_list, json_obj_from_text

class Reflextion_prompt_builder:
    def __init__(self):
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''
        self.action_history = ''
        self.inventory = ''
        self.current_environment = ''
        self.action_history = ''
        self.action_list = ''
        self.another_room_info = ''
        self.last_action = ''
        self.last_obs = ''

    def build(self):
        system_msg = ''
        system_msg += '\n'
        system_msg += f'You are a text game player, and you just made an incorrect action ({self.last_action}->{self.last_obs}). Please analyze the reason for the mistake based on the information below and response in json form.\n'
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Action history: {self.action_history}\n'
        user_msg += f'Inventory: {self.inventory}\n'
        user_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        user_msg += f'Current environment: {self.current_environment}\n' if self.current_environment else ''
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += 'Reflection (in json form: {"reflection": string}):\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

    def sys_usr_msg(self):
        return self.system_msg, self.user_msg
    

def action_obs_pairs_to_history_with_reflection(action_obs_pairs):
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

class Builder_full_action_list_with_reflextion(Builder_old_style):
    def action_obs_pairs_to_history(self, action_obs_pairs):
        return action_obs_pairs_to_history_with_reflection(action_obs_pairs)

def quest_reflextion(system_msg,
                        user_msg,
                        gpt_type=GPT4OMINI,
                        verbose = False):
    text = quest_simple_get_text(system_msg, user_msg, gpt_type, verbose)
    print('REFLECTION!!!!!!!!!!!!!!')
    print(text)
    obj = json_obj_from_text(text)
    return obj['reflection']


# 2024.11.20 Reflextion
class GPT_agent_smooth_desc_another_room_reflextion(GPT_caller_simplify_desc_only_smooth_another_room):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflextion_promptor = Reflextion_prompt_builder()
        self.reflextion_log = ''

    def try_adjust_and_execute(self, command): # 2024.11.20
        if command is None:
            return self.env.act(command, no_augment=self.no_augment)
        else:
            available_actions = self.env.env.available_actions;
            if command not in available_actions:
                self.update_action_history_myself(command, 'The command is not in the available command list, please recheck.')
                return None, None, None, None, {'raw_obs': 'The command is not in the available command list, please recheck.', 'command_not_in_list': True, 'env_act_failed': True};
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

    def llm_reflextion(self):
        self.reflextion_promptor.action_history = action_obs_pairs_to_history_with_reflection(self.action_obs_pairs)
        self.reflextion_promptor.last_action = self.action_obs_pairs[-1][0]
        self.reflextion_promptor.last_obs = self.action_obs_pairs[-1][1]
        self.reflextion_promptor.inventory = self.inventory
        self.reflextion_promptor.another_room_info = self.another_room_info
        self.reflextion_promptor.action_list = actions_to_list(self.available_actions)
        self.reflextion_promptor.current_environment = self.desc
        self.reflextion_promptor.build()
        sys, usr = self.reflextion_promptor.sys_usr_msg()
        self.reflextion_log += self.reflextion_promptor.prompt
        respsonse = quest_reflextion(sys, usr)
        self.reflextion_log += respsonse
        return respsonse
    
    def update_action_history_with_reflextion(self, reflextion):
        self.env.append_command_obs_pair('reflection', reflextion)
        self.set_act_result_to_body(action_obs_pairs=self.env.get_action_obs_pair())

    def update_action_history_myself(self, act, obs):
        self.env.append_command_obs_pair(act, obs)
        self.set_act_result_to_body(action_obs_pairs=self.env.get_action_obs_pair())

    def reflect(self, wront_command, extra_info):
        self.env.append_command_obs_pair(wront_command, extra_info['raw_obs'])
        print('反思反思反思')
        reflextion = self.llm_reflextion()
        self.update_action_history_with_reflextion(reflextion)
        extra_info['reflexted'] = True;

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
            # print(extra_info_raw)
            extra_info = extra_info | extra_info_raw
            if extra_info['env_act_failed']: # 指令执行失败的情况，需要让模型反思，随后调用模型获取下一个指令，NOTE: 该情况下步数不会增加
                self.step_counter += 1 # 就算是失败也要增加步数
                # NOTE: Reflextion: 让模型反思该错误并把结果放进action history
                self.action_obs_pairs = action_obs_pairs if action_obs_pairs else self.action_obs_pairs
                self.reflect(command, extra_info)
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
                    if extra_info['is_placing_item'] and extra_info['placing_failed']:
                        self.action_obs_pairs = action_obs_pairs if action_obs_pairs else self.action_obs_pairs
                        self.reflect(command, extra_info)
                        action_obs_pairs = self.action_obs_pairs
                    # NOTE: 由于步数增加意味着环境发生改变，需要更新环境描述
                    self.update_observation(description, inventory, available_actions, action_obs_pairs)
                    command_may_none, raw_extra_info = self.call_llm_get_next_command_try_twice()
                    extra_info = extra_info | raw_extra_info
                    return command_may_none, extra_info
        else:
            print(f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}')
            self.save()
            extra_info = extra_info | {'reach_step_limit': True, 'raw_obs': f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}'}
            return None, extra_info
        
    def act_until_error(self, command=None):
        next_command, extra_info = self.act_and_call(command)
        while (not extra_info['reach_step_limit']) and (not extra_info['won']):
            next_command, extra_info = self.act_and_call(next_command)
            if next_command is None:
                print('XXXXXXXXXXXXX 通过LLM获取下一个指令错误，请检查extra_info:')
                # print(extra_info)
                return self
            

    def save_hook(self):
        super().save_hook()
        env = self.env.env
        filename = 'exp/auto_filename/' + self.filename_raw + f'score_{env.last_reward}_reflextion_log.txt'
        f = open(filename, 'w')
        f.write(self.reflextion_log)
        f.close()