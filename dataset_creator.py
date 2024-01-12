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
    # TODO: 处理错位问题
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

