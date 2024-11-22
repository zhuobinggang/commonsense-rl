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
    return available_action_text


def action_obs_pairs_to_history(action_obs_pairs):
    # print(action_obs_pairs)
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} -> {obs} '
    else:
        action_history = 'No action was taken now.'
    return action_history

def considerations_to_text(considerations):
    if len(considerations) == 0:
        return 'Nothing.'
    text = ''
    idx = 1
    for consideration in considerations:
        text += f'{idx}. {consideration}\n'
        idx += 1
    return '\n' + text.strip()