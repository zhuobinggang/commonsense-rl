import global_variable as G
from openai import OpenAI
import pyperclip
client = OpenAI()

class Prompt_builder:
    def __init__(self, task = G.TASK, action_history = None, inventory = None, current_enviroment = None, example = None,  action_list = None, question = G.QUESTION, consideration = G.FILL_IN_TEMPLATE, next_action = G.FILL_IN_TEMPLATE, another_room_info = None):
        self.task = task
        self.action_history = action_history
        self.inventory = inventory
        self.current_enviroment = current_enviroment
        self.example = example
        self.action_history = action_history
        self.action_list = action_list
        self.question = question
        self.consideration = consideration
        self.next_action = next_action
        self.another_room_info = another_room_info
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''
    def build(self):
        system_msg = ''
        system_msg += f'Task: {self.task}\n' if self.task else ''
        system_msg += f'Example walkthrough: {self.example}\n' if self.example else ''
        system_msg += f'Action history: {self.action_history}\n' if self.action_history else ''
        system_msg += f'Inventory: {self.inventory}\n' if self.inventory else ''
        system_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        system_msg += f'Current enviroment: {self.current_enviroment}\n' if self.current_enviroment else ''
        system_msg = system_msg.strip()
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Action you can take:\n{self.action_list}\n' if self.action_list else ''
        user_msg += f'Question: {self.question}\n' if self.question else ''
        user_msg += f'Consideration: {self.consideration}\n' if self.consideration else ''
        user_msg += f'Next action: {self.next_action}\n' if self.next_action else ''
        user_msg = user_msg.strip()
        self.user_msg = user_msg
        self.prompt = f'{system_msg}\n{user_msg}'


class Builder1: # 2024.8.9之前的
    def __init__(self):
        pass
    def build(self, current_enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, cot = True, one_shot_easy = False, no_augment = False):
        builder = Prompt_builder()
        builder.inventory = inventory
        builder.current_enviroment = current_enviroment
        available_action_text = ''
        for act in available_actions:
            available_action_text += f'* {act}\n'
        builder.action_list = available_action_text
        action_history = ''
        if len(action_obs_pairs) > 0:
            for idx, (act, obs) in enumerate(action_obs_pairs):
                action_history += f'Action {idx}: {act} -> {obs} '
        else:
            action_history = 'No action was taken now.'
        builder.action_history = action_history
        if zero_shot:
            builder.example = None
        else: # one shot
            if one_shot_easy and no_augment:
                builder.example = G.ONE_SHOT_EXP_SIMPLE_NO_AUG
            elif one_shot_easy and not no_augment:
                builder.example = G.ONE_SHOT_EXP_AUGMENT_SIMPLE
            elif not one_shot_easy and no_augment:
                builder.example = G.ONE_SHOT_EXP_NO_AUG
            elif not one_shot_easy and not no_augment: # 提案手法
                builder.example = G.ONE_SHOT_EXP_AUGMENT
        if cot:
            builder.question = G.QUESTION
        else:
            builder.question = G.QUESTION_NO_COT
            builder.consideration = None
        self.builder = builder 
        self.builder.build()

    def sys_usr_msg(self):
        return self.builder.system_msg, self.builder.user_msg


def quest_gpt_raw(system_msg, user_msg, gpt_type):
    completion = client.chat.completions.create(
      model=gpt_type, # 
      messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
      ],
      temperature = 0
    )
    # To clipboard
    content = completion.choices[0].message.content
    usage = str(completion.usage)
    # To clipboard
    # NOTE: 自动提取command
    copied = False
    for line in content.split('\n'):
        line = line.lower()
        if line.startswith('next action:'):
            text_to_paste = line.replace('next action:', '').strip()
            pyperclip.copy(f"c.act_and_call('{text_to_paste}')")
            print(f'COMMAND GOT: {text_to_paste}')
            copied = True
    if not copied:
        pyperclip.copy('')
        print(f'BE CAREFULL!')
    dic = {'response': content, 'usage': usage}
    return completion, dic

class GPT_Caller:
    def __init__(self, env, zero_shot = True, gpt_type = 'gpt-3.5-turbo-0613',  cot = True, one_shot_easy = False, no_augment = False, step_limit = 20, builder = None):
        self.zero_shot = zero_shot
        self.gpt_type = gpt_type
        self.env = env
        self.cot =cot 
        self.one_shot_easy = one_shot_easy
        self.no_augment = no_augment
        self.step_limit = step_limit 
        self.filename = f'ZERO_SHOT_{zero_shot}_COT_{cot}_GPT_{gpt_type}_ONE_SHOT_EASY_{one_shot_easy}_NO_AUGMENT_{no_augment}_STEP_LIMIT_{step_limit}_{env.env.meta_info}.pkl' 
        self.filename_raw = f'ZERO_SHOT_{zero_shot}_COT_{cot}_GPT_{gpt_type}_ONE_SHOT_EASY_{one_shot_easy}_NO_AUGMENT_{no_augment}_STEP_LIMIT_{step_limit}_{env.env.meta_info}' 
        print(self.filename)
        # print(f'ZERO SHOT: {zero_shot}')
        # print(f'COT: {cot}')
        # print(f'GPT VERSION: {gpt_type}')
        # print(f'ONE_SHOT_EASY: {one_shot_easy}')
        # print(f'NO_AUGMENT: {no_augment}')
        # print(f'STEP_LIMIT: {step_limit}')
        self.step_counter = 0
        # Add 2024.8.9
        self.builder = builder or Builder1()
    def __call__(self, description, inventory, available_actions, action_obs_pairs):
        self.builder.build(description, inventory, available_actions, action_obs_pairs, zero_shot = self.zero_shot, cot = self.cot, one_shot_easy = self.one_shot_easy, no_augment = self.no_augment)
        system_msg, user_msg = self.builder.sys_usr_msg()
        dd, dic = quest_gpt_raw(system_msg, user_msg, gpt_type = self.gpt_type)
        if self.env is not None:
            self.env.env.system_user_msgs.append(system_msg + user_msg)
            self.env.env.gpt_responses.append(dd)
            self.env.env.readable_log += (system_msg + user_msg + '\n\n\n' + dic['response'] + '\n\n\n\n')
        return dd
    def act_and_call(self, command = None): # if None, a new begining
        if self.step_counter < self.step_limit:
            self.step_counter += 1
            description, inventory, available_actions, action_obs_pairs = self.env.act(command, no_augment = self.no_augment) # 2024.8.9 修复bug？好像step这一步没有考虑augmentation的问题
            if self.env.env.end:
                print('YOU WIN! NO API CALL NEED.')
                self.save()
            else:
                self.env.env.score_by_step.append(self.env.env.last_reward)
                _ = self.__call__(description, inventory, available_actions, action_obs_pairs)
        else:
            print(f'NO MORE ACTION CAN TAKE, STEP_COUNTER NOW: {self.step_counter}')
            self.save()
    def save(self):
        env = self.env.env
        filename = 'exp/auto_filename/' + self.filename_raw + f'score_{env.last_reward}.txt'
        f = open(filename, 'w')
        f.write(env.readable_log)
        f.close()
        # 处理错位问题
        # dic = {'env_meta': env.meta_info, 'system_user_msgs': env.system_user_msgs, 'gpt_responses': env.gpt_responses}
        # import pickle
        # with open(filename, 'wb') as handle:
        #     pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def log(self):
        print(self.env.env.readable_log)
