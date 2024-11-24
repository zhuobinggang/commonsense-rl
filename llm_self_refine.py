# Self refine until the action be the same.
import global_variable as G
from llm_simplify import GPT4OMINI, quest_simple_get_text, get_client, quest_gpt_simple
from common import json_obj_from_text
from llm_consideration_buffer import GPT_caller_react

class Prompt_builder_self_refine:

    def __init__(self):
        self.task = G.TASK_NEW
        self.action_history = ''
        self.inventory = ''
        self.another_room_info = ''
        self.current_enviroment = ''
        self.action_list = ''
        # last consideration and action
        self.prev_consideration = ''
        self.prev_action = ''
        # others
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''

    def build(self):
        system_msg = ''
        system_msg += f'Task: {self.task}\n' if self.task else ''
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Action history: {self.action_history}\n' if self.action_history else ''
        user_msg += f'Inventory: {self.inventory}\n' if self.inventory else ''
        user_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        user_msg += f'Environment: {self.current_enviroment}\n' if self.current_enviroment else ''
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        # NOTE: New things
        user_msg += f'Your consideration: {self.prev_consideration}\n' if self.prev_consideration else ''
        user_msg += f'Your action choice: {self.prev_action}\n' if self.prev_action else ''
        user_msg += 'Does your action choice need improvement? You have a chance to refine it. Please answer with your consideration and your new action choice (in json form: {"consideration": string, "action": string}):'
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

def quest_self_refine(system_msg,
                        user_msg,
                        gpt_type=GPT4OMINI,
                        verbose = False):
    text = quest_simple_get_text(system_msg, user_msg, gpt_type, verbose)
    print('SELF REFINE!!!!!!!!!!!!!!')
    obj = json_obj_from_text(text)
    consideration = obj['consideration']
    action = obj['action']
    return consideration, action

class GPT_caller_self_refine(GPT_caller_react):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refine_promptor = Prompt_builder_self_refine()

    def quest_my_llm(self, system_msg, user_msg, llm_type, verbose = False):
        complete, dic, _ = quest_gpt_simple(get_client(), system_msg,
                                        user_msg,
                                        llm_type, verbose=verbose) # 获得the_command，可能为空
        self.add_to_readable_log(str(dic['response']))
        # NOTE: self refine
        self.refine_promptor.action_history = self.builder.builder.action_history
        self.refine_promptor.inventory = self.builder.builder.inventory
        self.refine_promptor.another_room_info = self.builder.builder.another_room_info
        self.refine_promptor.current_enviroment = self.builder.builder.current_enviroment
        self.refine_promptor.action_list = self.builder.builder.action_list
        self.refine_promptor.prev_consideration = dic['response']['consideration']
        self.refine_promptor.prev_action = dic['response']['action']
        self.refine_promptor.build()
        consideration, action = quest_self_refine(self.refine_promptor.system_msg, self.refine_promptor.user_msg)
        self.add_to_readable_log(f'\n\nSELF-REFINE:\nconsideration: {consideration}\naction: {action}\n\n')
        dic['response']['consideration'] = consideration
        dic['response']['action'] = action
        self.llm_response_dic_got(dic) # NOTE: BUG 
        return complete, dic, action