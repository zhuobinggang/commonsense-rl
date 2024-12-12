from functools import lru_cache
TRAIN_PATH = '/home/taku/Downloads/cog2019_ftwp/games/train'

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

@lru_cache(maxsize=None)
def train_set_v0():
    file_paths = filenames_good_length()[:10]
    file_paths = [file.replace('.json', '.z8') for file in file_paths]
    return file_paths