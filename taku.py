### 让游戏给出所有可能命令的方式
def run():
    from textworld import EnvInfos
    from games import dataset
    infos_to_request = EnvInfos(description=True, inventory=True, admissible_commands=True,won=True, lost=True,location = True,
            last_action=True,game=True,facts=True,entities=True)
    game_path = "/home/taku/research/commonsense-rl/games/twc/easy/train/tw-iqa-cleanup-objects1-take1-rooms1-train-M32pu02bS65MUBxV.ulx"
    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, 50)
    obs, infos = env.reset()
    return infos['admissible_commands'][0]


def printer1(obs, infos):
    print(f"{obs} \n\nNow choose 'one' command from {infos['admissible_commands'][0]}\n")

def get_default_env(printer = printer1):
    from textworld import EnvInfos
    from games import dataset
    infos_to_request = EnvInfos(description=True, inventory=True, admissible_commands=True,won=True, lost=True,location = True,
            last_action=True,game=True,facts=True,entities=True)
    game_path = "/home/taku/research/commonsense-rl/games/twc/easy/train/tw-iqa-cleanup-objects1-take1-rooms1-train-M32pu02bS65MUBxV.ulx"
    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, 50)
    obs, infos = env.reset()
    if printer:
        printer(obs[0], infos)
    return env, obs[0], infos

def step(env, cmd, printer = printer1):
    obs, reward, is_terminal, infos = env.step([cmd])
    if printer:
        printer(obs[0], infos)
    return obs[0], infos


