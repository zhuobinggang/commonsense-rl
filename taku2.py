# All you need is here
from neighbor_room_info import Caller_Neighbor , get_game_env
from taku import get_game_env_by_path
from common import get_time_str
import global_variable as G

class Caller(Caller_Neighbor):
    def __init__(self, env, gpt_type = G.GPT4o, zero_shot = False, cot = True, one_shot_easy = False, no_augment = False, disable_another_room = True, step_limit = 20):
        super().__init__(env, zero_shot, gpt_type, cot, one_shot_easy, no_augment, step_limit, disable_another_room)
        self.filename = f'ANOTHER_ROOM_{not disable_another_room}_ZERO_SHOT_{zero_shot}_COT_{cot}_GPT_{gpt_type}_ONE_SHOT_EASY_{one_shot_easy}_NO_AUGMENT_{no_augment}_STEP_LIMIT_{step_limit}_{env.meta_info}_{get_time_str()}.pkl' 

def run(game_idx = 0):
    env = get_game_env(2, game_idx)
    caller = Caller(env, step_limit = 20, gpt_type = G.GPT4, zero_shot = True)
    caller(None) # first step
    return caller

def run_disscussion():
    env = get_game_env_by_path('/home/taku/research/zhuobinggang/commonsense-rl/exp/handmade_game_exp/games/taku_handmade_env1.ulx' )
    env.meta_info = 'dd'
    caller = Caller(env, step_limit = 20, gpt_type = G.GPT35)
    caller(None) # first step
    return caller
