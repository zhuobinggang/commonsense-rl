from openai import OpenAI
from taku_step_by_step import get_game_env, ONE_SHOT_EXP, printer_command_with_index
import global_variable as G
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
            pyperclip.copy(f"c('{text_to_paste}')")
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


def get_system_user_msg_builder(current_enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, cot = True, one_shot_easy = False, no_augment = False):
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
    return builder

def get_system_user_msg_v2(current_enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, cot = True, one_shot_easy = False, no_augment = False):
    builder = get_system_user_msg_builder(current_enviroment, inventory, available_actions, action_obs_pairs, zero_shot, cot, one_shot_easy, no_augment)
    builder.build()
    return builder.system_msg, builder.user_msg


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
