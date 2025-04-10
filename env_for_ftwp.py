from env import Env_extra_info, infos_to_request

def get_all_games(target_path, recipe = 1):
    import os
    # 检查路径是否存在
    if not os.path.isdir(target_path):
        raise ValueError("目标路径不存在或不是文件夹")
    # 遍历文件夹并过滤出符合条件的文件
    matching_files = [
        filename for filename in os.listdir(target_path)
        if filename.startswith(f"tw-cooking-recipe{recipe}") and filename.endswith(".z8")
    ]
    return matching_files

class Env_ftwp(Env_extra_info):

    def __init__(self,
                 recipe_num = 1,
                 game_index = 0,
                 no_augment=True):
        # 初始化
        self.PATH_PREFIX = '/home/taku/Downloads/cog2019_ftwp/games/test/'
        self.DATASET_NAMES = ['train', 'test', 'valid']
        self.recipe_num = recipe_num
        self.game_index = game_index
        self.env = self.get_game_env(recipe_num=recipe_num, game_index=game_index)
        # act
        self.RIGHT_POSITION_HINT = 'Right position.'
        self.WRONG_POSITION_HINT = 'Wrong position, you should put it somewhere else, maybe the other room.'
        # 反馈强化
        self.no_augment = no_augment

    def is_won(self, info):
        return info['won']
    
    def is_lost(self, info):
        return info['lost']
    
    def get_desc(self, info):
        return ' '.join(info['description'].strip().split())
    
    def get_inventory(self, info):
        return ' '.join(info['inventory'].strip().split())
    
    def get_available_actions(self, info):
        return info['admissible_commands']

    def origin_env_step(self, cmd):
        obs, reward, is_not_terminal, info = self.env.step(cmd)
        self.info = info
        obs = ' '.join(obs.split()) # 去掉换行符
        return obs, reward, is_not_terminal, info    

    def get_moves(self, info):
        return info['moves']
    
    def get_score(self, info):
        return info['score']
    
    def get_obs(self, obs_raw):
        return ' '.join(obs_raw.strip().split())

    def get_game_env(self, recipe_num=0, game_index=0, dataset_index = 1, need_reset = True):
        files = get_all_games(self.PATH_PREFIX, recipe=recipe_num) # ['tw-cooking-recipe1+take1+cook+cut+open+go6-8MN7Cv1vS2epHEqe.z8']
        file = files[game_index]
        import textworld.gym
        env_id = textworld.gym.register_game(self.PATH_PREFIX + file, infos_to_request, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        if need_reset:
            obs, infos = env.reset()
        return env    

class Env_ftwp_by_path(Env_ftwp):
    def __init__(self,
                 game_path,
                 no_augment=True):
        # 初始化
        self.game_path = game_path
        self.env = self.get_game_env(game_path)
        # act
        self.RIGHT_POSITION_HINT = 'Right position.'
        self.WRONG_POSITION_HINT = 'Wrong position, you should put it somewhere else, maybe the other room.'
        # 反馈强化
        self.no_augment = no_augment

    def get_game_env(self, path, need_reset = True):
        import textworld.gym
        env_id = textworld.gym.register_game(path, infos_to_request, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        if need_reset:
            obs, infos = env.reset()
            self.info = infos
        return env
    

def test():
    env = Env_ftwp()
    _ = env.act()
    return env

def test_get_train_set():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0()
    return Env_ftwp_by_path(file_paths[0])