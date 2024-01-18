from openai import OpenAI
from taku_step_by_step import get_game_env, ONE_SHOT_EXP, printer_command_with_index
import taku_step_by_step as taku
import pyperclip
client = OpenAI()
from dataset_creator import DEFAULT_SYSTEM_PROMPT

def quest_gpt4(system_msg, user_msg, gpt3 = False):
    if gpt3:
        model="gpt-3.5-turbo"
    else:
        model = "gpt-4-1106-preview"
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
      ]
    )
    print(completion.choices[0].message.content)
    print(completion.usage)
    # To clipboard
    text_to_paste = f'{completion.choices[0].message.content}\n{completion.usage}'
    pyperclip.copy(text_to_paste)
    return completion


def call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True):
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
    return quest_gpt4(system_msg, user_msg, gpt3 = True)
    # return f'{system_msg}\n{user_msg}'
    

# 流程： taku.act_step_by_step负责基础的字符串处理+引擎交互+结果存储
# call_gpt4_and_print负责将上一步的结果汇总成prompt并向openai发送请求
def act_step_by_step(env, command = None, caller = call_gpt4_and_print):
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
