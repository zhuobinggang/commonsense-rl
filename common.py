def load_dataset(filename):
    import pickle
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def get_time_str():
    from datetime import datetime
    # get time now
    dt = datetime.now()
    # format it to a string
    return dt.strftime('%Y%m%d_%H%M%S')

def action_history_to_reward_by_step(act_string):
    acts = act_string.split('Action ')
    acts = acts[2:]
    assert acts[0].startswith('0')
    score = 0
    scores = []
    for idx, act in enumerate(acts):
        if act.endswith('Right position. '):
            score += 1
        scores.append(score)
    return scores

def filename_to_reward_by_step(filename):
    dd = load_dataset(filename)
    dd = dd['system_user_msgs'][-1]
    hist = dd.split('\n')[2]
    return action_history_to_reward_by_step(hist)

def json_obj_from_text(text):
    import re
    import json
    text = text.replace('\n','')
    pattern = r'\{.+\}'
    result = re.search(pattern, text)
    try:
        json_string = result.group(0)
        json_data = json.loads(json_string)
        print(json_data)
        return json_data
    except IndexError:
        print("No json found!")
        return None
    
def actions_to_list(actions):
    available_action_text = ''
    for act in actions:
        available_action_text += f'* {act}\n'
    return available_action_text.strip()

def actions_to_list_number(actions, start_index = 0):
    available_action_text = ''
    count = start_index
    for act in actions:
        available_action_text += f'{count} {act}\n'
        count += 1
    return available_action_text.strip()

def action_obs_pairs_to_history(action_obs_pairs, seperator = '->'):
    # print(action_obs_pairs)
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} {seperator} {obs} '
    else:
        action_history = 'No action was taken now.'
    return action_history


def action_obs_pairs_to_history_react_style(action_obs_pairs, considerations):
    if len(action_obs_pairs) == 0:
        return ''
    text = ''
    idx = 1
    for consideration, (action, obs) in zip(considerations, action_obs_pairs):
        text += f'Consideration {idx}: {consideration}\n'
        text += f'Action {idx}: {action}\n'
        text += f'Observation {idx}: {obs}\n'
        idx += 1
    return '\n' + text.strip()



def considerations_to_text(considerations):
    if len(considerations) == 0:
        return 'Nothing.'
    text = ''
    idx = 1
    for consideration in considerations:
        text += f'{idx}. {consideration}\n'
        idx += 1
    return '\n' + text.strip()


def training_line_prepare(sys, usr, agent):
    obj = {'messages': [{"role": "system", "content": sys.strip()}, {"role": "user", "content": usr.strip()}, {"role": "assistant", "content": agent.strip()}]}
    return obj



def calculate_normalized_score_by_path(folder_path = 'exp/auto_filename'):
    import os
    import re
    """
    Calculate the normalized score from filenames in the given folder.
    Filenames should end in the format: 'scoreX_of_Y.txt'.

    Args:
        folder_path (str): The path to the folder containing the files.

    Returns:
        float: The normalized score (sum of scores / sum of maximum scores).
    """
    # Regular expression to match and capture scores and max scores
    pattern = re.compile(r'score(\d+)_of_(\d+)\.txt')

    total_score = 0
    total_max_score = 0

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            # Extract the score and maximum score from the filename
            score = int(match.group(1))
            max_score = int(match.group(2))
            total_score += score
            total_max_score += max_score

    # Avoid division by zero
    if total_max_score == 0:
        return 0.0

    # Calculate and return the normalized score
    return total_score / total_max_score



def extract_room_name(description):
    import re
    """
    从字符串中提取房间名。

    :param description: 包含房间名的字符串
    :return: 提取的房间名，如果未找到则返回 None
    """
    match = re.search(r"-= (.*?) =-", description)
    if match:
        return match.group(1)
    return None


def is_recipe_feedback(feedback):
    return feedback.startswith('You open the copy of')


def extract_recipe(text):
    """
    从给定的文本中提取 'Ingredients:' 后面的内容。
    
    参数:
    text (str): 包含完整配方的字符串。

    返回:
    str: 从 'Ingredients:' 开始到末尾的部分。如果未找到 'Ingredients:'，返回空字符串。
    """
    # 找到 "Ingredients:" 的位置
    marker = "Ingredients:"
    index = text.find(marker)
    
    # 如果找到了 "Ingredients:"
    if index != -1:
        # 返回从 "Ingredients:" 后面开始的内容
        return text[index:].strip()
    else:
        # 如果没有找到，返回空字符串
        return ""