from openai import OpenAI
from taku_step_by_step import get_game_env, ONE_SHOT_EXP, printer_command_with_index
import taku_step_by_step as taku
import pyperclip
client = OpenAI()
from dataset_creator import DEFAULT_SYSTEM_PROMPT

def quest_gpt(system_msg, user_msg, gpt_type):
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
    print(content)
    print(usage)
    # To clipboard
    # NOTE: 自动提取command
    copied = False
    for line in content.split('\n'):
        line = line.lower()
        if line.startswith('next action:'):
            text_to_paste = line.replace('next action:', '').strip()
            pyperclip.copy(text_to_paste)
            print(f'COMMAND GOT: {text_to_paste}')
            copied = True
    if not copied:
        pyperclip.copy('')
        print(f'BE CAREFULL!')
    dic = {'response': content, 'usage': usage}
    return completion

def quest_gpt4(system_msg = 'You are a helpful assistant.', user_msg = 'Tell me a joker.', gpt3 = False):
    if gpt3:
        model="gpt-3.5-turbo"
    else:
        # model = "gpt-4-1106-preview"
        model = "gpt-4-0125-preview"
    return quest_gpt(system_msg, user_msg, model)


class GPT_Caller:
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
    def __call__(self, description, inventory, available_actions, action_obs_pairs):
        # system_msg, user_msg = get_system_user_msg(description, inventory, available_actions, action_obs_pairs = action_obs_pairs, zero_shot = self.zero_shot)
        system_msg, user_msg = get_system_user_msg_v2(description, inventory, available_actions, action_obs_pairs, zero_shot = self.zero_shot, cot = self.cot, one_shot_easy = self.one_shot_easy, no_augment = self.no_augment)
        dd = quest_gpt(system_msg, user_msg, gpt_type = self.gpt_type)
        if self.env is not None:
            self.env.system_user_msgs.append(system_msg + user_msg)
            self.env.gpt_responses.append(dd)
        return dd
    def step(self, command):
        if self.step_counter < self.step_limit:
            self.step_counter += 1
            description, inventory, available_actions, action_obs_pairs = taku.act_step_by_step_obs_augment(self.env, command)
            if self.env.end:
                print('YOU WIN! NO API CALL NEED.')
                self.save()
            else:
                self.env.score_by_step.append(self.env.last_reward)
                _ = self.__call__(description, inventory, available_actions, action_obs_pairs)
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

def get_system_user_msg(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True):
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
    print(action_history)
    if zero_shot:
        system_msg = f"""Task: {task}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}"""
    else:
        system_msg = f"""Task: {task}
Example walkthrough: {ONE_SHOT_EXP}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}"""
    user_msg = f"""Action you can take: 
{available_action_text}
Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>"""
    return system_msg, user_msg

################### taku added 2024.5.22 START ###################

ONE_SHOT_EXP_AUGMENT = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point.Right position. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point.Right position. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point.Right position. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point.Right position. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point.Right position. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point.Right position. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point.Right position."""


ONE_SHOT_EXP_NO_AUG = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point."""

ONE_SHOT_EXP_AUGMENT_SIMPLE = """Action 0: take dirty gray underpants from work table -> You take the dirty gray underpants from the work table. Action 1: insert dirty gray underpants into washing machine -> You put the dirty gray underpants into the washing machine.Your score has just gone up by one point.Right position."""


ONE_SHOT_EXP_SIMPLE_NO_AUG = """Action 0: take dirty gray underpants from work table -> You take the dirty gray underpants from the work table. Action 1: insert dirty gray underpants into washing machine -> You put the dirty gray underpants into the washing machine.Your score has just gone up by one point."""

QUESTION_TEMPLATE_NO_COT = """Question: To put things in there proper locations and improve your score, what should you do? Choose 'one' action from above list.
Next action: <fill in>"""

QUESTION_TEMPLATE = """Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>"""

def get_system_user_msg_v2(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, cot = True, one_shot_easy = False, no_augment = False):
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


################### taku added 2024.5.22 END #######################

def call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True):
    system_msg, user_msg = get_system_user_msg(enviroment, inventory, available_actions, action_obs_pairs, zero_shot = zero_shot)
    return quest_gpt4(system_msg, user_msg, gpt3 = False)
    # return f'{system_msg}\n{user_msg}'
    

# 流程： taku.act_step_by_step负责基础的字符串处理+引擎交互+结果存储
# call_gpt4_and_print负责将上一步的结果汇总成prompt并向openai发送请求
def act_step_by_step(env, command = None):
    caller = GPT_Caller(zero_shot = True, gpt_type = 'gpt-3.5-turbo-0613')
    return taku.act_step_by_step(env, command, caller)

### 20240115 use finetuned gpt3.5
def quest_finetuned_gpt35(prompt):
    model = "ft:gpt-3.5-turbo-1106:tottori-university-nlp-lab::8hBX82zF"
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
      ]
    )
    print(completion.choices[0].message.content)
    print(completion.usage)
    # To clipboard
    text_to_paste = f'{completion.choices[0].message.content}\n{completion.usage}'
    pyperclip.copy(text_to_paste)
    dic = {'response': completion.choices[0].message.content, 'usage': str(completion.usage)}
    return dic


def call_finetuned_gpt35_and_print(enviroment, inventory, available_actions, action_obs_pairs = []):
    prompt = printer_command_with_index(enviroment, inventory, available_actions, action_obs_pairs)
    prompt = prompt.strip() + '\nNext action: '
    print(prompt)
    response_dic = quest_finetuned_gpt35(prompt)
    return prompt, response_dic

def act_step_by_step_finetuned_model(env, command_idx = -1, caller = call_finetuned_gpt35_and_print):
    if not hasattr(env, 'record'): # 2024.1.16 增加env.record
        env.record = []
    command = None if command_idx == -1 else env.available_actions[command_idx]
    gpt_response = taku.act_step_by_step(env, command, caller)
    if gpt_response is not None:
        x, y = gpt_response
        env.record.append((x,y))

def save_record(env, filename = None):
    if not filename:
        filename = 'exp/finetuned_model_walkthrough/' + env.meta_name + '.pickle'
    dic = {'env_meta': env.meta_info, 'record': env.record}
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

##### 1.20 finetuned gpt35 with action explanation ####


def quest_finetuned_gpt35_with_action_explanation(prompt):
    model = "ft:gpt-3.5-turbo-1106:tottori-university-nlp-lab::8iggnTFZ"
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
      ]
    )
    print(completion.choices[0].message.content)
    print(completion.usage)
    # To clipboard
    text_to_paste = f'{completion.choices[0].message.content}\n{completion.usage}'
    pyperclip.copy(text_to_paste)
    dic = {'response': completion.choices[0].message.content, 'usage': str(completion.usage)}
    return dic

def quest_finetuned_gpt35_with_action_explanation_hard_only(prompt):
    model = "ft:gpt-3.5-turbo-1106:tottori-university-nlp-lab::8j0lH4Xt"
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
      ]
    )
    print(completion.choices[0].message.content)
    print(completion.usage)
    # To clipboard
    text_to_paste = f'{completion.choices[0].message.content}\n{completion.usage}'
    pyperclip.copy(text_to_paste)
    dic = {'response': completion.choices[0].message.content, 'usage': str(completion.usage)}
    return dic
