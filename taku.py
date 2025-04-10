from env import infos_to_request

### 让游戏给出所有可能命令的方式
def run():
    from games import dataset
    game_path = "/home/taku/research/commonsense-rl/games/twc/easy/train/tw-iqa-cleanup-objects1-take1-rooms1-train-M32pu02bS65MUBxV.ulx"
    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, 50)
    obs, infos = env.reset()
    return infos['admissible_commands'][0]


def printer1(obs, infos):
    print(f"{obs[0]} \nNow choose 'one' command from {infos['admissible_commands'][0]}")

def get_default_env(printer = printer1):
    from games import dataset
    game_path = "/home/taku/research/commonsense-rl/games/twc/easy/train/tw-iqa-cleanup-objects1-take1-rooms1-train-M32pu02bS65MUBxV.ulx"
    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, 50)
    obs, infos = env.reset()
    if printer:
        printer(obs[0], infos)
    return env, obs[0], infos

def step(env, cmd, caller = printer1, need_reward = False):
    obs, reward, is_not_terminal, infos = env.step([cmd])
    if caller:
        caller(obs, infos)
    if not is_not_terminal:
        print('\n\nGAME OVER!')
    env.info = infos # Added 2024.5.23
    return (obs, infos) if not need_reward else (obs, infos, reward)

LEVELS = ['easy', 'medium', 'hard']
DATASET_NAMES = ['train', 'test', 'valid']
def get_game_name(level_index = 0, game_index = 0, dataset_index = 1):
    import glob
    level_name = LEVELS[level_index]
    dataset_name = DATASET_NAMES[dataset_index]
    GAME_NAMES = glob.glob(f'/home/taku/research/zhuobinggang/commonsense-rl/games/twc/{level_name}/{dataset_name}/*ulx')
    game_name = GAME_NAMES[game_index]
    print(game_name)
    return game_name

def get_game_env(level_index = 0, game_index = 0, dataset_index = 1):
    game_path = get_game_name(level_index, game_index, dataset_index)
    return get_game_env_by_path(game_path)


def get_game_env_by_path(game_path, need_reset = True):
    from games import dataset
    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, max_episode_steps = 50, mode = 'test')
    if need_reset:
        obs, infos = env.reset()
    return env

