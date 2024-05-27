from gpt4 import quest_gpt, get_game_env, get_system_user_msg_v2, get_system_user_msg_builder
import taku_step_by_step as taku
from global_variable import * 

#### 保存相邻房间的信息 ####
#### TODO: 直接填加一个字段叫做Another room:
#### TODO: 需要找到判断房间移动的准确根据, 大概从env里面可以找到？
#### 把step之后的info也保存到env里面，可能有重要信息 -> ['last_action', 'inventory', 'description', 'moves', 'entities', 'won', 'max_score', 'admissible_commands', 'location', 'lost', 'facts', 'score'] -> 没有想要的地理位置信息
class GPT_Caller_Easy_Call:
    def __init__(self, env, zero_shot = True, gpt_type = 'gpt-3.5-turbo-0613',  cot = True, one_shot_easy = False, no_augment = False, step_limit = 20):
        self.zero_shot = zero_shot
        self.gpt_type = gpt_type
        self.env = env
        self.cot =cot 
        self.one_shot_easy = one_shot_easy
        self.no_augment = no_augment
        self.step_limit = step_limit 
        self.filename = f'ZERO_SHOT_{zero_shot}_COT_{cot}_GPT_{gpt_type}_ONE_SHOT_EASY_{one_shot_easy}_NO_AUGMENT_{no_augment}_STEP_LIMIT_{step_limit}_{env.meta_info}.pkl' 
        print(self.filename)
        # print(f'ZERO SHOT: {zero_shot}')
        # print(f'COT: {cot}')
        # print(f'GPT VERSION: {gpt_type}')
        # print(f'ONE_SHOT_EASY: {one_shot_easy}')
        # print(f'NO_AUGMENT: {no_augment}')
        # print(f'STEP_LIMIT: {step_limit}')
        self.step_counter = 0
    def call_gpt(self, description, inventory, available_actions, action_obs_pairs):
        # system_msg, user_msg = get_system_user_msg(description, inventory, available_actions, action_obs_pairs = action_obs_pairs, zero_shot = self.zero_shot)
        system_msg, user_msg = get_system_user_msg_v2(description, inventory, available_actions, action_obs_pairs, zero_shot = self.zero_shot, cot = self.cot, one_shot_easy = self.one_shot_easy, no_augment = self.no_augment)
        dd = quest_gpt(system_msg, user_msg, gpt_type = self.gpt_type)
        if self.env is not None:
            self.env.system_user_msgs.append(system_msg + user_msg)
            self.env.gpt_responses.append(dd)
        return dd
    def __call__(self, command):
        if self.step_counter < self.step_limit:
            self.step_counter += 1
            description, inventory, available_actions, action_obs_pairs = taku.act_step_by_step_obs_augment(self.env, command, not self.no_augment)
            if self.env.end:
                print('YOU WIN! NO API CALL NEED.')
                self.save()
            else:
                self.env.score_by_step.append(self.env.last_reward)
                _ = self.call_gpt(description, inventory, available_actions, action_obs_pairs)
        else:
            print(f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}')
            self.save()
    def save(self):
        filename = 'exp/auto_filename/' + self.filename
        env = self.env
        # 处理错位问题
        dic = {'env_meta': env.meta_info, 'system_user_msgs': env.system_user_msgs, 'gpt_responses': env.gpt_responses}
        import pickle
        with open(filename, 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Caller_Neighbor(GPT_Caller_Easy_Call):
    def __init__(self, env, zero_shot = True, gpt_type = 'gpt-3.5-turbo-0613',  cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, disable_another_room = False):
        super().__init__(env, zero_shot, gpt_type, cot, one_shot_easy, no_augment, step_limit)
        self.last_desc = ''
        self.another_room_info = 'Unknown'
        self.filename = f'ANOTHER_ROOM_{not disable_another_room}_ZERO_SHOT_{zero_shot}_COT_{cot}_GPT_{gpt_type}_ONE_SHOT_EASY_{one_shot_easy}_NO_AUGMENT_{no_augment}_STEP_LIMIT_{step_limit}_{env.meta_info}.pkl' 
        self.disable_another_room = disable_another_room
        print(f'ANOTHER ROOM INFO: {not self.disable_another_room}')
    def call_gpt(self, description, inventory, available_actions, action_obs_pairs):
        # system_msg, user_msg = get_system_user_msg(description, inventory, available_actions, action_obs_pairs = action_obs_pairs, zero_shot = self.zero_shot)
        if self.disable_another_room: # added 2024.5.24 to disable the another room info
            system_msg, user_msg = get_system_user_msg_v2(description, inventory, available_actions, action_obs_pairs, zero_shot = self.zero_shot, cot = self.cot, one_shot_easy = self.one_shot_easy, no_augment = self.no_augment) 
        else: # DEFAULT
            system_msg, user_msg = get_system_user_msg_v3(description, inventory, available_actions, action_obs_pairs, zero_shot = self.zero_shot, cot = self.cot, one_shot_easy = self.one_shot_easy, no_augment = self.no_augment, another_room_info=self.another_room_info) # NOTE: changed get_system_user_msg v2 to v3
        dd = quest_gpt(system_msg, user_msg, gpt_type = self.gpt_type)
        if self.env is not None:
            self.env.system_user_msgs.append(system_msg + '\n' + user_msg)
            self.env.gpt_responses.append(dd)
        return dd
    def __call__(self, command):
        if self.step_counter < self.step_limit:
            self.step_counter += 1
            if command and command.startswith('go '): # NOTE: record last room info
                self.another_room_info = self.last_desc
                print('UPDATED ANOTHER ROOM INFO')
            description, inventory, available_actions, action_obs_pairs = taku.act_step_by_step_obs_augment(self.env, command, not self.no_augment)
            self.last_desc = description # NOTE
            if self.env.end:
                print('YOU WIN! NO API CALL NEED.')
                self.save()
            else:
                self.env.score_by_step.append(self.env.last_reward)
                _ = self.call_gpt(description, inventory, available_actions, action_obs_pairs)
        else:
            print(f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}')
            self.save()


def get_system_user_msg_v3(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, cot = True, one_shot_easy = False, no_augment = False, another_room_info = 'Unknown'):
    builder = get_system_user_msg_builder(enviroment, inventory, available_actions, action_obs_pairs, zero_shot, cot, one_shot_easy, no_augment)
    builder.another_room_info = another_room_info
    builder.build()
    return builder.system_msg, builder.user_msg

def run(game_idx = 0):
    env = get_game_env(2, game_idx)
    # 提案手法
    # caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-4o-2024-05-13', cot = True, one_shot_easy = False, no_augment = False, step_limit = 30)
    # -GPT4o
    # caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-4-1106-preview', cot = True, one_shot_easy = False, no_augment = False, step_limit = 30) # 差最后两个环境没试。不过暂时先做优先级更高的
    # caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-4-1106-preview', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20) 
    # caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-4o-2024-05-13', cot = False, one_shot_easy = False, no_augment = False, step_limit = 20, disable_another_room = True)  # NOTE: 关闭另一房间信息
    # caller = Caller_Neighbor(env, zero_shot = True, gpt_type = 'gpt-4o-2024-05-13', cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, disable_another_room = True)  # NOTE: 关闭另一房间信息
    caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-4o-2024-05-13', cot = True, one_shot_easy = False, no_augment = True, step_limit = 20, disable_another_room = True)  # NOTE: 关闭另一房间信息
    # # -GPT4
    # caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-3.5-turbo-0613', cot = True, one_shot_easy = False, no_augment = False)
    # # -one-shot 
    # # TODO: 需要检查prompt
    # caller = Caller_Neighbor(env, zero_shot = True, gpt_type = 'gpt-4o-2024-05-13', cot = True, one_shot_easy = False, no_augment = False)
    # # -COT
    # # NOTE: 检查prompt OK
    # caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-4o-2024-05-13', cot = False, one_shot_easy = False, no_augment = False, step_limit = 30)
    # # -FA
    # # TODO: 需要首先支持将Another room选项关闭
    # # TODO: 需要检查prompt
    # caller = Caller_Neighbor(env, zero_shot = False, gpt_type = 'gpt-4o-2024-05-13', cot = False, one_shot_easy = False, no_augment = False)
    caller(None) # first step
    return caller

