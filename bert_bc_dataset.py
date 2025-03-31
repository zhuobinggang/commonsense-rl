# 复刻cogni的ner模型
import os
import pandas as pd
from ftwp_info import all_game_paths
from tqdm import tqdm
import textworld.gym
from textworld import EnvInfos, gym
import common

TRAIN_PATH = '/home/taku/Downloads/cog2019_ftwp/games/train'
TEST_PATH = '/home/taku/Downloads/cog2019_ftwp/games/test'
VALID_PATH = '/home/taku/Downloads/cog2019_ftwp/games/valid'

DEBUG = False


requested_infos = EnvInfos(description=True, inventory=True,
                            admissible_commands=True, objective=True,
                            verbs=True, command_templates=True,
                            entities=True, max_score=True, won=True,
                            lost=True, extras=["recipe", "walkthrough"])

def get_cv_games(path, split = 'train'):
    return all_game_paths(f'{path}/{split}')


def get_game_name(gamepath):
    return os.path.split(gamepath)[-1]

def extract_walkthrough_dataset(games):
    """
    runs a sequence of games and collects all information into a dataframe
    """
    gamesteps = []
    for game in tqdm(games):
        gamesteps += run_game(game)
    return pd.DataFrame(gamesteps)


class Walkthrough:
    def __init__(self, walkthrough, include_cookbook=True):
        if include_cookbook:
            self.walkthrough = self.filter_walkthrough(walkthrough)
        else:
            self.walkthrough = walkthrough
        self.pos = 0
        self.include_cookbook = include_cookbook
        self.doors = ['plain door', 'sliding patio door', 'front door',
                      'commercial glass door', 'sliding door', 'patio door',
                      'fiberglass door', 'barn door', 'screen door',
                      'wooden door', 'frosted glass door',
                      'frosted-glass door']

    def filter_walkthrough(self, walkthrough):
        # remove open door
        walkthrough = [cmd for cmd in walkthrough
                       if not (cmd.startswith('open')
                               and cmd.endswith('door'))]
        return walkthrough

    def get_next_command(self, description, admissible_commands,
                         seen_cookbook):
        if self.include_cookbook:
            return self.get_next_command_cookbook(description,
                                                  admissible_commands,
                                                  seen_cookbook)
        else:
            return self.get_next_command_basic(admissible_commands,
                                               seen_cookbook)

    def get_next_command_basic(self, admissible_commands, seen_cookbook):
        if self.pos < len(self.walkthrough):
            cmd = self.walkthrough[self.pos]
            self.pos += 1
            return cmd
        else:
            return None

    def get_first_closed_door(self, description):
        for door in self.doors:
            check = f'closed {door}'
            if check in description:
                return door
        return ''

    def get_next_command_cookbook(self, description, admissible_commands,
                                  seen_cookbook):
        # 意思就是，如果有门就先把所有门打开……有什么意义？
        if not seen_cookbook and 'examine cookbook' in admissible_commands: # 如果还没调用过且有exmanine cookbook这个命令，就返回这个命令
            return 'examine cookbook'
        elif self.get_first_closed_door(description): # 看过cookbook，遍历所有的door，找到第一扇关着的门，返回打开这个门的命令
            ent = self.get_first_closed_door(description)
            return f'open {ent}'
        elif self.pos < len(self.walkthrough):
            cmd = self.walkthrough[self.pos]
            self.pos += 1
            return cmd
        else:
            return None

def extra_info_by_game_path(game_path):
    import json
    json_path = game_path.replace('.z8', '.json')
    f = open(json_path)
    data = json.load(f)
    f.close()
    return data['extras']

def run_game(gamefile):
    """ runs a game following the walkthrough and extracts the information """
    if DEBUG:
        print('Game name:', gamefile)
    env_id = textworld.gym.register_games([gamefile], requested_infos)
    env = gym.make(env_id)
    obs, infos = env.reset()
    wts = common.filter_commands_default(infos['extra.walkthrough'])
    walk = Walkthrough(wts) # 这个walkthrough本身也会对命令进行过滤 -> 不是过滤，而是莫名其妙地，如果有门就先把门打开
    if DEBUG:
        print('Walk:', walk.walkthrough)
    seen_cookbook = False
    gamename = get_game_name(gamefile)
    gamesteps = []
    idx = 0
    # taku
    taku_extra = extra_info_by_game_path(gamefile)
    taku_recipe = taku_extra['recipe']
    while True:
        cmd = walk.get_next_command(infos['description'],
                                    infos['admissible_commands'],
                                    seen_cookbook) # 相当于一个walkthrough修正器，比如可能exmanine cookbook没有出现在walkthrough中…
        if cmd is None:
            break
        if DEBUG:
            print('cmd:', cmd)
        step = {
            'gamename': gamename,
            'step': idx,
            'description': infos['description'],
            'inventory': infos['inventory'],
            'recipe': taku_recipe if seen_cookbook else '', # NOTE: 不知道为什么现在info取不到recipe，所以用taku_recipe来代替
            'admissible_commands': infos['admissible_commands'], # ?
            'entities': infos['entities'],
            'verbs': infos['verbs'],
            'command_templates': infos['command_templates'], # 这个居然可以直接取？？？
            'objective': infos['objective'],
            'max_score': infos['max_score'],
            # 'has_won': infos['has_won'],
            'won': infos['won'],
            # 'has_lost': infos['has_lost'],
            'lost': infos['lost'],
            'walkthrough': infos['extra.walkthrough'], # 好像可以直接取到，那就留着他把
            'seen_cookbook': seen_cookbook,
            'command': cmd
        }
        obs, scores, dones, infos = env.step(cmd)
        step['reward'] = scores
        step['obs'] = obs
        if DEBUG:
            print('obs:', obs)
        gamesteps.append(step)
        if cmd == 'examine cookbook':
            seen_cookbook = True
        idx += 1

    assert dones # ?
    return gamesteps


def extract_datasets(datapath, outputpath, need_valid = True, need_train = True):
    """ runs all games and saves dataframes with state/command """
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    walkext_dict, sufix = None, ''
    if need_train:
        train_games = get_cv_games(datapath, 'train')
        train_data = extract_walkthrough_dataset(train_games)
        train_data.to_csv(os.path.join(outputpath,
                        f'walkthrough_train{sufix}.csv'), index=False)
    if need_valid:
        valid_games = get_cv_games(datapath, 'valid')
        valid_data = extract_walkthrough_dataset(valid_games)
        valid_data.to_csv(os.path.join(outputpath,
                        f'walkthrough_valid{sufix}.csv'), index=False)


# DONE: 2025.3.24
def generate_dataset():
    output = 'exp/auto_filename'
    datapath = '/home/taku/Downloads/cog2019_ftwp/games'
    extract_datasets(datapath, output, need_valid=True, need_train=False)
    # fix_recipe(output) # 将未查看cookbook的步骤的recipe删掉
    # generate_datasets_commands(output)