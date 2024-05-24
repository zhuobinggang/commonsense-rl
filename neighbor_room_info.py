from gpt4o import *
from gpt4 import quest_gpt, GPT_Caller
import taku_step_by_step as taku

#### 保存相邻房间的信息 ####
#### TODO: 直接填加一个字段叫做Another room:
#### TODO: 需要找到判断房间移动的准确根据, 大概从env里面可以找到？
#### 把step之后的info也保存到env里面，可能有重要信息 -> ['last_action', 'inventory', 'description', 'moves', 'entities', 'won', 'max_score', 'admissible_commands', 'location', 'lost', 'facts', 'score'] -> 没有想要的地理位置信息

class Caller_Neighbor(GPT_Caller):
    def __init__(self, env, zero_shot = True, gpt_type = 'gpt-3.5-turbo-0613',  cot = True, one_shot_easy = False, no_augment = False, step_limit = 20):
        super().__init__(env, zero_shot, gpt_type, cot, one_shot_easy, no_augment, step_limit)
        self.last_desc = ''
        self.another_room_info = 'Unknown'
    def __call__(self, description, inventory, available_actions, action_obs_pairs):
        # system_msg, user_msg = get_system_user_msg(description, inventory, available_actions, action_obs_pairs = action_obs_pairs, zero_shot = self.zero_shot)
        system_msg, user_msg = get_system_user_msg_v3(description, inventory, available_actions, action_obs_pairs, zero_shot = self.zero_shot, cot = self.cot, one_shot_easy = self.one_shot_easy, no_augment = self.no_augment, self.another_room_info) # NOTE: changed get_system_user_msg v2 to v3
        dd = quest_gpt(system_msg, user_msg, gpt_type = self.gpt_type)
        if self.env is not None:
            self.env.system_user_msgs.append(system_msg + user_msg)
            self.env.gpt_responses.append(dd)
        return dd
    def step(self, command):
        if self.step_counter < self.step_limit:
            self.step_counter += 1
            if command.startswith('go '): # NOTE: record last room info
                self.another_room_info = self.last_desc
                print('UPDATED ANOTHER ROOM INFO')
            description, inventory, available_actions, action_obs_pairs = taku.act_step_by_step_obs_augment(self.env, command)
            self.last_desc = description # NOTE
            if self.env.end:
                print('YOU WIN! NO API CALL NEED.')
                self.save()
            else:
                self.env.score_by_step.append(self.env.last_reward)
                _ = self.__call__(description, inventory, available_actions, action_obs_pairs)
        else:
            print(f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}')
            self.save()


# Added 2024.5.23
# Add: another room info
def get_system_user_msg_v3(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, cot = True, one_shot_easy = False, no_augment = False, another_room_info = 'Unknown'):
    task = 'You are a experienced text game player, your goal is put things in there proper locations and improve your score.'
    available_action_text = ''
    for act in available_actions:
        available_action_text += f'* {act}\n'
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} -> {obs} '
    else:
        action_history = 'No action was taken now.'
    if zero_shot:
        system_msg = f"""Task: {task}
Action history: {action_history}
Inventory: {inventory}
Another room: {another_room_info}
Current enviroment: {enviroment}"""
    else:
        if no_augment:
            walkthrough = ONE_SHOT_EXP_SIMPLE_NO_AUG if one_shot_easy else ONE_SHOT_EXP_NO_AUG
        else: 
            walkthrough = ONE_SHOT_EXP_AUGMENT_SIMPLE if one_shot_easy else ONE_SHOT_EXP_AUGMENT
        system_msg = f"""Task: {task}
Example walkthrough: {walkthrough}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}"""
    question_template = QUESTION_TEMPLATE if cot else QUESTION_TEMPLATE_NO_COT
    user_msg = f"""Action you can take: 
{available_action_text}
{question_template}"""
    return system_msg, user_msg
