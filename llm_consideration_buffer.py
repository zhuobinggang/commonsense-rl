# 简单地维护一个consideration buffer好了
from llm_reflextion import GPT_agent_smooth_desc_another_room_reflextion
from common import action_obs_pairs_to_history, considerations_to_text, action_obs_pairs_to_history_react_style
from llm_simplify import Prompt_builder
from llm_simplify_smooth_summarization import GPT_caller_simplify_desc_only_smooth_another_room

class Builder_with_consideration_buffer:

    def __init__(self):
        self.builder = None
        self.name = 'Builder_with_consideration_buffer'

    def action_obs_pairs_to_history(self, action_obs_pairs):
        return action_obs_pairs_to_history(action_obs_pairs)

    def build(self,
              current_enviroment,
              inventory,
              available_actions,
              action_obs_pairs=[],
              another_room_info = '',
              recent_considerations = []):
        builder = Prompt_builder()
        builder.inventory = inventory
        builder.current_enviroment = current_enviroment
        available_action_text = ''
        for act in available_actions:
            available_action_text += f'* {act}\n'
        builder.action_list = available_action_text
        builder.action_history = self.action_obs_pairs_to_history(action_obs_pairs)
        # another room info
        builder.another_room_info = another_room_info
        # 2024.11.22 consideration buffer
        builder.recent_considerations = considerations_to_text(recent_considerations)
        self.builder = builder
        self.builder.build()

    def sys_usr_msg(self):
        return self.builder.system_msg, self.builder.user_msg

class GPT_caller_consideration_buffer(GPT_agent_smooth_desc_another_room_reflextion):
    def __init__(self, *args, **kwargs):
        GPT_caller_simplify_desc_only_smooth_another_room.__init__(self, *args, **kwargs)
        self.builder = Builder_with_consideration_buffer()
        self.consideration_buffer = []
        self.consideration_buffer_size = 3

    def reflect(self, wrong_command, extra_info):
        return # No reflect
    
    def llm_response_dic_got(self, dic):
        self.consideration_buffer.append(dic['response']['consideration'])
        if len(self.consideration_buffer) > self.consideration_buffer_size:
            self.consideration_buffer = self.consideration_buffer[1:]
    
    def build_prompt(self, desc, inventory, available_actions, act_obs_pairs):
        self.builder.build(desc,
                    inventory,
                    available_actions,
                    act_obs_pairs, recent_considerations= self.consideration_buffer)
        
    def save_hook(self):
        GPT_caller_simplify_desc_only_smooth_another_room.save_hook(self)


class Builder_react_style:

    def __init__(self):
        self.builder = None
        self.name = 'Builder_react_style'

    def action_obs_pairs_to_history(self, action_obs_pairs, considerations):
        return action_obs_pairs_to_history_react_style(action_obs_pairs, considerations)

    def build(self,
              current_enviroment,
              inventory,
              available_actions,
              action_obs_pairs=[],
              another_room_info = '',
              considerations = []):
        builder = Prompt_builder()
        builder.inventory = inventory
        builder.current_enviroment = current_enviroment
        available_action_text = ''
        for act in available_actions:
            available_action_text += f'* {act}\n'
        builder.action_list = available_action_text
        builder.action_history = self.action_obs_pairs_to_history(action_obs_pairs, considerations)
        # another room info
        builder.another_room_info = another_room_info
        self.builder = builder
        self.builder.build()

    def sys_usr_msg(self):
        return self.builder.system_msg, self.builder.user_msg

class GPT_caller_react(GPT_caller_consideration_buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder = Builder_react_style()

    def llm_response_dic_got(self, dic):
        self.consideration_buffer.append(dic['response']['consideration'])
    
    def build_prompt(self, desc, inventory, available_actions, act_obs_pairs):
        self.builder.build(desc,
                    inventory,
                    available_actions,
                    act_obs_pairs, 
                    another_room_info = self.another_room_info,
                    considerations= self.consideration_buffer)
        
    def save_hook(self):
        GPT_caller_simplify_desc_only_smooth_another_room.save_hook(self)