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
    print(completion.choices[0].message.content)
    print(completion.usage)
    # To clipboard
    text_to_paste = f'{completion.choices[0].message.content}\n{completion.usage}'
    pyperclip.copy(text_to_paste)
    return completion

def quest_gpt4(system_msg = 'You are a helpful assistant.', user_msg = 'Tell me a joker.', gpt3 = False):
    if gpt3:
        model="gpt-3.5-turbo"
    else:
        # model = "gpt-4-1106-preview"
        model = "gpt-4-0125-preview"
    return quest_gpt(system_msg, user_msg, model)


class GPT_Caller:
    def __init__(self, zero_shot = True, gpt_type = 'gpt-3.5-turbo-0613', env = None):
        self.zero_shot = zero_shot
        self.gpt_type = gpt_type
        self.env = env
        print(f'ZERO SHOT: {zero_shot}')
        print(f'GPT VERSION: {gpt_type}')
        if env is not None:
            print(f'ENV SETTED!')
    def __call__(self, description, inventory, available_actions, action_obs_pairs):
        system_msg, user_msg = get_system_user_msg(description, inventory, available_actions, action_obs_pairs = action_obs_pairs, zero_shot = self.zero_shot)
        dd = quest_gpt(system_msg, user_msg, gpt_type = self.gpt_type)
        if self.env is not None:
            self.env.system_user_msgs.append(system_msg + user_msg)
            self.env.gpt_responses.append(dd)
        return dd

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
