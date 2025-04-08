import common_new as common
import textworld.gym
from textworld import EnvInfos, gym
from functools import lru_cache

# 重新实现game
def init_env(game_file):
    requested_infos = EnvInfos(description=True, inventory=True,
                               admissible_commands=True, objective=False,
                               # verbs=True, command_templates=True,
                               entities=True, max_score=True, won=True, score=True,
                               moves = True,
                               lost=True, extras=["walkthrough"]) # 注意，取不到recipe，只能从obs中获取
    env_id = textworld.gym.register_games([game_file], requested_infos)
    env = gym.make(env_id)
    return env


class Game:
    def __init__(self, game_path):
        self.game_path = game_path
        self.env = init_env(game_path)
        self.obs, self.info = None, None
        self.reward, self.done = 0, False
    def reset(self):
        self.obs, self.info = self.env.reset()
        return self.obs, self.info
    def act(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        return self.obs, self.reward, self.done, self.info


class Fake_model:
    def __init__(self):
        self.counter = 0
    def predict(self, obs, info):
        # 这里需要根据obs和info来选择动作
        # 这里简单返回一个随机动作
        action = info['extra.walkthrough'][self.counter]
        self.counter += 1
        if self.counter >= len(info['extra.walkthrough']):
            self.counter = 0
        return action

def test_game(game: Game, model = Fake_model()):
    obs, info = game.reset()
    counter = 0
    while counter < 100:
        counter += 1
        action = model.predict(obs, info)
        obs, reward, done, info = game.act(action)
        if done:
            break
    return info

# ============

class Game_with_history(Game):
    def __init__(self, game_path):
        super().__init__(game_path)
        self.action_obs_pairs = []
    def act(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        self.action_obs_pairs.append((action, self.obs))
        return self.obs, self.reward, self.done, self.info
    
class Game_handle_recipe(Game_with_history):
    def __init__(self, game_path):
        super().__init__(game_path)
        self.recipe_raw = ''
        self.recipe = ''
        self.obs_raw = ''
    def act(self, action):
        self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
        obs = self.obs_raw
        # obs simplify
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
        self.action_obs_pairs.append((action, obs)) # 不在这里处理，而是放到game_state中处理
        self.obs = obs
        return self.obs, self.reward, self.done, self.info
    
def default_game():
    return Game_handle_recipe('/home/taku/Downloads/cog2019_ftwp/games/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')

@lru_cache(maxsize=128) # 一个episode最多为100步，因此128足够了
def clean_action_obs(action, obs):
    ACT, OBS = action, obs
    if action == 'examine cookbook' and common.is_recipe_feedback(obs):
        OBS = 'recipe got!'
    elif common.is_description_feedback(obs):
        room_name = common.extract_room_name(obs)
        OBS = f'you entered {room_name}.'
    return ACT, OBS

class Game_state:
    def __init__(self):
        self.room = ''
        self.description_raw = ''
        self.recipe_raw = ''
        self.inventory_raw = ''
        self.action_obs_pairs = []
        self.admissible_commands = []
        self.act_obs_clean_cache = {}
    def recipe_clean(self):
        return common.extract_recipe(self.recipe_raw, need_clean=True)
    def inventory_clean(self):
        return common.handle_inventory_text(self.inventory_raw)
    def description_clean(self):
        return common.description_simplify(self.description_raw)
    def clean_action_obs_pairs(self, action_obs_pairs):
        return [clean_action_obs(action, obs) for action, obs in action_obs_pairs]
    def action_history(self, history_window = 100, seperator='>', no_action_text=''):
        action_obs_pairs = self.clean_action_obs_pairs(self.action_obs_pairs)
        action_history_text = common.action_obs_pairs_to_history(action_obs_pairs, seperator=seperator, no_action_text=no_action_text, history_window = history_window)        
        return action_history_text
    def available_commands_clean(self):
        return common.actions_to_list_number(common.filter_commands_default(self.admissible_commands))

def game_state_from_game(game: Game_handle_recipe):
    state = Game_state()
    state.room = common.extract_room_name(game.info['description'])
    state.description_raw = game.info['description']
    state.recipe_raw = game.recipe_raw
    state.inventory_raw = game.info['inventory']
    state.action_obs_pairs = game.action_obs_pairs
    state.admissible_commands = game.info['admissible_commands']
    return state

def bert_prompt_from_game_state(game_state: Game_state, need_action_history = True, history_window = 100):
    x = ''
    x += f"Room: {game_state.room}\n"
    x += f"Recipe: {game_state.recipe_clean()}\n"
    x += f"Inventory: {game_state.inventory_clean()}\n" # NOTE: 2025.3.18 增加以平等测试
    if need_action_history:
        action_history_text = game_state.action_history(history_window=history_window)
        x += f"Action history: {action_history_text}\n" 
    available_commands_text = game_state.available_commands_clean()
    x += f'Available actions:\n{available_commands_text}\n'
    return x

def test():
    game = default_game()
    _ = game.reset()
    game.act('go east')
    game.act('examine cookbook')
    print(bert_prompt_from_game_state(game_state_from_game(game)))