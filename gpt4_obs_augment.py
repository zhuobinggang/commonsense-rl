from openai import OpenAI
from taku_step_by_step import get_game_env, ONE_SHOT_EXP, act_step_by_step_obs_augment
import pyperclip
client = OpenAI()

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
    content = completion.choices[0].message.content
    usage = str(completion.usage)
    print(content)
    print(usage)
    # To clipboard
    # NOTE: 自动提取command
    for line in content.split('\n'):
        line = line.lower()
        if line.startswith('next action:'):
            text_to_paste = line.replace('next action:', '').strip()
            pyperclip.copy(text_to_paste)
            print(f'COMMAND GOT: {text_to_paste}')
    dic = {'response': content, 'usage': usage}
    return dic

#### 2024.1.25 ablation expriment
def call_gpt4_and_print_sub_COT(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, gpt3 = False):
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
Question: To put things in there proper locations and improve your score, what should you do? Choose 'one' action from above list.
Next action: <fill in>"""
    x = system_msg + '\n' + user_msg
    y = quest_gpt4(system_msg, user_msg, gpt3 = gpt3)
    return x, y


#### 2024.1.25 ablation expriment
def call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, gpt3 = False):
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
    x = system_msg + '\n' + user_msg
    y = quest_gpt4(system_msg, user_msg, gpt3 = gpt3)
    return x, y
    

def interact_with_env_raw(env, command = None, gpt3 = False):
    if not hasattr(env, 'dataset'): # 2024.1.8 增加env.dataset
        env.dataset = []
    enviroment, inventory, available_actions, action_obs_pairs = act_step_by_step_obs_augment(env, command)
    if not hasattr(env, 'end'):
        x, y = call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs, zero_shot = True, gpt3 = gpt3)
        env.dataset.append((x,y))

def interact_with_env(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = False)

def interact_with_env_gpt3(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = True)

def interact_with_env_oneshot(env, command = None):
    if not hasattr(env, 'dataset'): # 2024.1.8 增加env.dataset
        env.dataset = []
    enviroment, inventory, available_actions, action_obs_pairs = act_step_by_step_obs_augment(env, command)
    if not hasattr(env, 'end'):
        x, y = call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs, zero_shot = False)
        env.dataset.append((x,y))

def save_dataset(env, filename = None):
    dataset = env.dataset
    if not filename:
        filename = 'exp/gpt4_zero_shot_obs_augment/' + env.meta_name + '.pickle'
    # 处理错位问题
    dic = {'env_meta': env.meta_info, 'dataset': dataset}
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_dataset_oneshot(env, filename = None):
    dataset = env.dataset
    if not filename:
        filename = 'exp/gpt4_one_shot_obs_augment/' + env.meta_name + '.pickle'
    # 处理错位问题
    dic = {'env_meta': env.meta_info, 'dataset': dataset}
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_dataset_gpt35(env, filename = None):
    dataset = env.dataset
    if not filename:
        filename = 'exp/gpt35_augment/' + env.meta_name + '.pickle'
    # 处理错位问题
    dic = {'env_meta': env.meta_info, 'dataset': dataset}
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset(filename):
    import pickle
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic
