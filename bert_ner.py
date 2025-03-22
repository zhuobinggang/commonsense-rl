# 复刻cogni的ner模型
import os
import pandas as pd
from ftwp_info import all_train_game_paths, all_test_game_paths
from tqdm import tqdm
import textworld.gym
from textworld import EnvInfos, gym

DEBUG = False

def get_game_name(gamepath):
    return os.path.split(gamepath)[-1]

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
        if not seen_cookbook and 'examine cookbook' in admissible_commands:
            return 'examine cookbook'
        elif self.get_first_closed_door(description):
            ent = self.get_first_closed_door(description)
            return f'open {ent}'
        elif self.pos < len(self.walkthrough):
            cmd = self.walkthrough[self.pos]
            self.pos += 1
            return cmd
        else:
            return None

def get_cv_games(path = None):
    if not path:
        return all_train_game_paths()
    else:
        return all_test_game_paths(path)


def run_game(gamefile, walkthrough_ext=None):
    """ runs a game following the walkthrough and extracts the information """
    if DEBUG:
        print('Game name:', gamefile)
    requested_infos = EnvInfos(description=True, inventory=True,
                               admissible_commands=True, objective=True,
                               verbs=True, command_templates=True,
                               entities=True, max_score=True, won=True,
                               lost=True, extras=["recipe", "walkthrough"])
    env_id = textworld.gym.register_games([gamefile], requested_infos)
    env = gym.make(env_id)
    obs, infos = env.reset()
    if walkthrough_ext:
        walk = Walkthrough(walkthrough_ext, include_cookbook=False)
    else:
        walk = Walkthrough(infos['extra.walkthrough'])
    if DEBUG:
        print('Walk:', walk.walkthrough)
    seen_cookbook = False
    gamename = get_game_name(gamefile)
    gamesteps = []
    idx = 0
    while True:
        cmd = walk.get_next_command(infos['description'],
                                    infos['admissible_commands'],
                                    seen_cookbook)
        if cmd is None:
            break
        if DEBUG:
            print('cmd:', cmd)
        step = {
            'gamename': gamename,
            'step': idx,
            'description': infos['description'],
            'inventory': infos['inventory'],
            'recipe': infos['extra.recipe'],
            'admissible_commands': infos['admissible_commands'],
            'entities': infos['entities'],
            'verbs': infos['verbs'],
            'command_templates': infos['command_templates'],
            'objective': infos['objective'],
            'max_score': infos['max_score'],
            # 'has_won': infos['has_won'],
            'won': infos['won'],
            # 'has_lost': infos['has_lost'],
            'lost': infos['lost'],
            'walkthrough': infos['extra.walkthrough'],
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

    assert dones
    return gamesteps


def extract_walkthrough_dataset(games, walkthrough_ext_dict=None):
    """
    runs a sequence of games and collects all information into a dataframe
    """
    gamesteps = []
    for game in tqdm(games):
        if walkthrough_ext_dict:
            walkext = walkthrough_ext_dict[get_game_name(game)]
        else:
            walkext = None
        gamesteps += run_game(game, walkext)
    return pd.DataFrame(gamesteps)


def extract_datasets(datapath, outputpath, use_walkthrough_ext=False):
    """ runs all games and saves dataframes with state/command """
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    if use_walkthrough_ext:
        gtr = pd.read_csv(os.path.join(outputpath, 'enhanced_walkthrough.csv'))
        gtr['walk_ext'] = gtr.walk_ext.apply(eval)
        walkext_dict = dict(zip(gtr.gamename.values, gtr.walk_ext.values))
        sufix = '_cookbook'
    else:
        walkext_dict, sufix = None, ''
    train_games = get_cv_games(datapath)
    train_data = extract_walkthrough_dataset(train_games, walkext_dict)
    train_data.to_csv(os.path.join(outputpath,
                      f'walkthrough_train{sufix}.csv'), index=False)



def generate_dataset():
    extract_datasets('/home/taku/Downloads/cog2019_ftwp/games/taku_test', 'exp/auto_filename')
    # enhanced_walkthrough(output)
    # extract_datasets(datapath, output, use_walkthrough_ext=True)
    # fix_recipe(output)
    # generate_datasets_commands(output)