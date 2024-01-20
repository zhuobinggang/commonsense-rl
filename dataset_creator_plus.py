# 性能を向上させるプロンプトの改善：Action Explaination
from dataset_creator import *

# 2024.1.8 : 将上一步的available action list保存在env里边，command用序号输入
def x_construct(enviroment, inventory, available_actions, action_obs_pairs = [], explanations = []):
    available_action_text = ''
    for idx, act in enumerate(available_actions):
        available_action_text += f'[{idx}] {act}\n'
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, ((act, obs), explanation),  in enumerate(zip(action_obs_pairs, explanations)):
            action_history += f'Action {idx}: {act} ({explanation}) -> {obs} '
    else:
        action_history = ''
    template = f"""Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}
Action explanation:
[A] Put item in proper place
[B] Take item from wrong place
[C] Take item that do not belong in this room
[D] Change room
Action you can take: 
{available_action_text}
"""
    # print(template)
    return template.strip() + '\nNext action: '

# 2024.1.18 
COMMAND_EXPLANATIONS = ['A', 'B', 'C', 'D', '?']
def interact_with_env_and_record_dataset(env, command_explanation_idx = -1, command_idx = -1):
    if not hasattr(env, 'dataset'): # 2024.1.8 增加env.dataset
        env.dataset = []
    if not hasattr(env, 'explanations'): # 2024.1.8 增加env.dataset
        env.explanations = []
    command_explanation = COMMAND_EXPLANATIONS[command_explanation_idx]
    if command_idx == -1:
        command = None
    else:
        command = env.available_actions[command_idx]
        env.explanations.append(command_explanation)
    enviroment, inventory, available_actions, action_obs_pairs = taku.act_step_by_step_simple(env, command)
    x = x_construct(enviroment, inventory, available_actions, action_obs_pairs, env.explanations)
    # x = taku.act_step_by_step_simple(env, command, printer)
    y = f'{command_explanation}-{command_idx}'
    env.dataset.append((x, y, env.instant_reward))
    print(x) # 打印发现

# 2024.1.20
def interact_with_env_and_record_walkthrough(env, response_dic = None):
    if not hasattr(env, 'dataset'): # 2024.1.8 增加env.dataset
        env.dataset = []
    if not hasattr(env, 'explanations'): # 2024.1.8 增加env.dataset
        env.explanations = []
    if response_dic is not None:
        y_from_gpt = response_dic['response']
        explanation, command_idx = y_from_gpt.split('-')
        env.explanations.append(explanation)
        command = env.available_actions[int(command_idx)]
    else:
        command = None
    enviroment, inventory, available_actions, action_obs_pairs = taku.act_step_by_step_simple(env, command)
    x = x_construct(enviroment, inventory, available_actions, action_obs_pairs, env.explanations)
    # x = taku.act_step_by_step_simple(env, command, printer)
    env.dataset.append((x, response_dic))
    print(x) # 打印发现
    return x

LEVEL_NAMES = ['easy', 'medium', 'hard']
def load_old_dataset(level, idx):
    level_name = LEVEL_NAMES[level]
    import pickle
    filename = f'exp/supervised_learning_dataset/train_{level_name}_{idx}.pickle' 
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def print_old_dataset(level, idx, need_x = True):
    print('XXXXXXXXXXXXXXXXXXXXXX')
    dic = load_old_dataset(level, idx)
    for x, y, reward in dic['dataset']:
        if need_x:
            print(x)
        print(f'LABEL: {y}\n\n')


def save_dataset(env, filename = None):
    dataset = env.dataset
    if not filename:
        filename = 'exp/supervised_dataset_plus/' + env.meta_name + '.pickle'
    # 处理错位问题
    dataset_fixed = []
    for i in range(len(dataset) - 1):
        x, _, _ = dataset[i]
        _, y, reward = dataset[i + 1]
        dataset_fixed.append((x, y, reward))
    dic = {'env_meta': env.meta_info, 'dataset': dataset_fixed}
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_walkthrough(env, filename = None):
    dataset = env.dataset
    if not filename:
        filename = 'exp/finetune_model_plus_walkthrough/' + env.meta_name + '.pickle'
    # 处理错位问题
    dataset_fixed = []
    for i in range(len(dataset) - 1):
        x, _ = dataset[i]
        _, y = dataset[i + 1]
        dataset_fixed.append((x, y))
    dic = {'env_meta': env.meta_info, 'dataset': dataset_fixed}
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


# shuffle and save dataset

def read_full_dataset():
    import glob
    pickles = 'exp/supervised_dataset_plus/*pickle'
    dataset_files = glob.glob(pickles)
    full_dataset = []
    for file in dataset_files:
        dic = load_dataset(file)
        full_dataset += dic['dataset']
    return full_dataset


def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def randomize_full_dataset(ds = None):
    if not ds:
        ds = read_full_dataset()
    import numpy as np
    np.random.seed(20240115)
    np.random.shuffle(ds)
    return ds

def save_randomized_full_dataset(ds = None):
    pass # NOTE: just create new instance every time


DEFAULT_SYSTEM_PROMPT = 'Select next action to improve your score.'
def create_json_row_dialogue(context, label):
    prompt = context.strip()
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(label)},
        ]
    }

def create_json_row_prompt_completion(context, label):
    prompt = context.strip()
    return {"prompt": prompt, "completion": str(label)}


def write_to_jsonl_dialogue_format(ds = None):
    if not ds:
        ds = randomize_full_dataset(None)
    import json
    with open("train_twc_player_dialogue.jsonl", "w") as f:
        for context, label, reward in ds:
            example_str = json.dumps(create_json_row_dialogue(context, label))
            f.write(example_str + "\n")


def write_to_jsonl_prompt_completion_format(ds = None):
    if not ds:
        ds = randomize_full_dataset(None)
    import json
    with open("train_twc_player_prompt_completion.jsonl", "w") as f:
        for context, label, reward in ds:
            example_str = json.dumps(create_json_row_prompt_completion(context, label))
            f.write(example_str + "\n")


# finetune
def finetune_gpt35_plus_explanation():
    from openai import OpenAI
    client = OpenAI()
    return client.fine_tuning.jobs.create(
            training_file="file-5tZvxws4npAmIHUkKTO8qMBD", # train_twc_player_dialogue_plus_explanation.jsonl
            model="gpt-3.5-turbo-1106",
            hyperparameters={"n_epochs": 3}
    )


##### 2024.1.20 test script example
def test_script_example():
    from dataset_creator_plus import interact_with_env_and_record_walkthrough, get_game_env, save_walkthrough
    from gpt4 import quest_finetuned_gpt35_with_action_explanation
    env = get_game_env(0,0)
    x = interact_with_env_and_record_walkthrough(env)
    y = quest_finetuned_gpt35_with_action_explanation(x)
    x = interact_with_env_and_record_walkthrough(env)
    y = quest_finetuned_gpt35_with_action_explanation(x)
    x = interact_with_env_and_record_walkthrough(env) # END
    save_walkthrough(env)

#### 2024.1.20 计算所有种类选项的占比，分析是否存在数据不均衡问题
def analyse_label_category_percentage():
    ds = randomize_full_dataset()
    dic = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for x, label, reward in ds:
        cate, command_idx = label.split('-')
        dic[cate] += 1
    return dic


