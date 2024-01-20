from taku_step_by_step import get_game_env, printer_command_with_index
import taku_step_by_step as taku

# 2024.1.8 : 将上一步的available action list保存在env里边，command用序号输入
def interact_with_env_and_record_dataset(env, command_idx = -1, printer = printer_command_with_index):
    if not hasattr(env, 'dataset'): # 2024.1.8 增加env.dataset
        env.dataset = []
    command = None if command_idx == -1 else env.available_actions[command_idx]
    x = taku.act_step_by_step(env, command, printer)
    y = command_idx
    env.dataset.append((x, y, env.instant_reward))
    print(x) # 打印发现


def save_dataset(env, filename = None):
    dataset = env.dataset
    if not filename:
        filename = 'exp/supervised_learning_dataset/' + env.meta_name + '.pickle'
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

def load_dataset(filename):
    import pickle
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic


########## 1.15整理数据集

def read_full_dataset():
    import glob
    pickles = 'exp/supervised_learning_dataset/*pickle'
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
    prompt = context.strip() + '\nNext action: '
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(label)},
        ]
    }

def create_json_row_prompt_completion(context, label):
    prompt = context.strip() + '\nNext action: '
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
