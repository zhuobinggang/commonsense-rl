from datetime import datetime

def load_dataset(filename):
    import pickle
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def get_time_str():
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

