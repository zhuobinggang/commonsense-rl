from functools import lru_cache
TRAIN_PATH = '/home/taku/Downloads/cog2019_ftwp/games/train'
TEST_PATH = '/home/taku/Downloads/cog2019_ftwp/games/test'
VALID_PATH = '/home/taku/Downloads/cog2019_ftwp/games/valid'

def read_json_files_in_directory(directory_path):
    import os
    import json
    """
    读取指定目录中的所有 JSON 文件，并返回一个包含所有 JSON 数据的字典列表。
    
    Args:
        directory_path (str): 文件夹路径。
        
    Returns:
        list[dict]: 包含 JSON 文件内容的字典列表。
    """
    json_data_list = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):  # 只处理 .json 文件
            file_path = os.path.join(directory_path, filename)
            try:
                # 打开并读取 JSON 文件内容
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    json_data_list.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"无法读取文件 {file_path}: {e}")
    
    return json_data_list

def read_walkthroughs_from_path(path):
    import json
    with open(path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data['extras']['walkthrough']
    

def walkthroughs_from_json_list(json_list):
    return [obj['extras']['walkthrough'] for obj in json_list]

def walkthrough_infos(walkthroughs):
    """
    计算 walkthroughs 中的最小步骤数、最大步骤数和平均步骤数。
    Args:
        walkthroughs (list[str]): 一个字符串列表，每个字符串代表一个 walkthrough。
    Returns:
        tuple: 包含最小步骤数、最大步骤数和平均步骤数的元组 (min_steps, max_steps, avg_steps)。
    """
    if not walkthroughs:
        return 0, 0, 0  # 如果列表为空，返回 0, 0, 0
    # 计算每个 walkthrough 的步骤数
    step_counts = [len(walkthrough) for walkthrough in walkthroughs]
    # 计算最小、最大和平均步骤数
    min_steps = min(step_counts)
    max_steps = max(step_counts)
    avg_steps = sum(step_counts) / len(step_counts)
    return min_steps, max_steps, avg_steps # 4, 62, 13.4


# @Results: {'go', 'drop', 'cook', 'open', 'take', 'recipe', 'cut'}
def skill_set(json_list):
    skills = set()
    for obj in json_list:
        skills.update(obj['metadata']['skills'].keys())
    return skills

@lru_cache(maxsize=None)
def filenames_good_length(directory_path = TRAIN_PATH):
    import os
    import json
    """
    读取指定目录中的所有 JSON 文件，并返回一个包含所有 JSON 数据的字典列表。
    
    Args:
        directory_path (str): 文件夹路径。
        
    Returns:
        list[dict]: 包含 JSON 文件内容的字典列表。
    """
    results = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):  # 只处理 .json 文件
            file_path = os.path.join(directory_path, filename)
            try:
                # 打开并读取 JSON 文件内容
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    length = len(data['extras']['walkthrough'])
                    if length > 10 and length < 20:
                        results.append(file_path)
            except (json.JSONDecodeError, IOError) as e:
                print(f"无法读取文件 {file_path}: {e}")
    return results


# 通过上面一个函数得到的最初10个游戏的技能表：{'recipe': 10, 'take': 5, 'cook': 3, 'cut': 7, 'open': 7, 'drop': 5, 'go': 7}
def filepaths_skill_count(paths):
    import json
    skill_dic = {}
    # 遍历文件夹中的所有文件
    for file_path in paths:
        try:
            # 打开并读取 JSON 文件内容
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                skills = list(data['metadata']['skills'].keys())
                for skill in skills:
                    if skill not in skill_dic:
                        skill_dic[skill] = 0
                    skill_dic[skill] += 1
        except (json.JSONDecodeError, IOError) as e:
            print(f"无法读取文件 {file_path}: {e}")
    return skill_dic

def train_set_v0(size = 10):
    import numpy as np
    file_paths = filenames_good_length()
    np.random.seed(0)
    file_paths = np.random.choice(file_paths, size, False)
    file_paths = [file.replace('.json', '.z8') for file in file_paths]
    return file_paths

@lru_cache(maxsize=None)
def all_train_game_paths():
    return all_game_paths(test_path = TRAIN_PATH)

@lru_cache(maxsize=None)
def all_valid_game_paths(shuffle = True):
    import numpy as np
    paths = all_game_paths(test_path = VALID_PATH)
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(paths)
    return paths

@lru_cache(maxsize=None)
def all_game_paths(test_path = TEST_PATH):
    import os
    results = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(test_path):
        if filename.endswith('.z8'):  # 只处理 .z8 文件
            file_path = os.path.join(test_path, filename)
            results.append(file_path)
    return results

def test_set_v0():
    import numpy as np
    paths = all_game_paths()
    np.random.seed(0)
    paths = np.random.choice(paths, 10, False)
    return paths


def extra_info_by_game_path(game_path):
    import json
    json_path = game_path.replace('.z8', '.json')
    f = open(json_path)
    data = json.load(f)
    f.close()
    return data['extras']

def walkthrough_by_game_path(game_path):
    return extra_info_by_game_path(game_path)['walkthrough']

def recipe_by_game_path(game_path):
    return extra_info_by_game_path(game_path)['recipe']

# 从valid set中获取临时的test集和valid集
def temp_test_valid_set():
    import numpy as np
    paths = all_game_paths(VALID_PATH)
    np.random.seed(0)
    paths = np.random.choice(paths, 25, False) # Only 5 games for valid to save money
    valid_paths = paths[:5]
    test_paths = paths[5:]
    return test_paths, valid_paths

# Date: 2024.12.21
# From: history.ftwp_4omini_finetune_base
def kitchen_not_found_games():
    test_paths, valid_paths = temp_test_valid_set()
    results = []
    results.append(test_paths[2])
    results.append(test_paths[6])
    results.append(test_paths[9])
    results.append(test_paths[11])
    results.append(test_paths[12])
    results.append(test_paths[13])
    results.append(test_paths[16])
    return results
        

def valid_set_v0():
    import numpy as np
    paths = all_game_paths(VALID_PATH)
    np.random.seed(0)
    paths = np.random.choice(paths, 5, False) # Only 5 games for valid to save money
    return paths


# ===================== 使用nltk来获取命令模板 ========================

# @result: 2025.1.8 确认了所有训练集中的命令以wold_list开头
# NOTE: all start words ['inventory', 'examine', 'open', 'take', 'drop', 'cook', 'slice', 'chop', 'dice', 'prepare', 'eat', 'go']):
# NOTE: examine cookbook and eat meal are only words start with examine and eat
def filter_commands(commands, word_list = ['inventory', 'open', 'take', 'drop', 'cook', 'slice', 'chop', 'dice', 'prepare', 'go']):
    """
    过滤出不以单词列表中的单词开头的指令。

    参数:
        commands (list): 包含指令的字符串列表。
        word_list (list): 包含要过滤的单词的列表。

    返回:
        list: 不以 word_list 中单词开头的指令列表。
    """
    available_actions = []
    for command in commands:
        if not any(command.startswith(word) for word in word_list):
            available_actions.append(command)
    return available_actions


def test_filter_commands(walkthroughs = None):
    if not walkthroughs:
        json_data_list = read_json_files_in_directory(TRAIN_PATH)
        walkthroughs = walkthroughs_from_json_list(json_data_list)
    commands = []
    for dd in walkthroughs:
        commands += dd
    return filter_commands(commands)
