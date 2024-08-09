# 重写一切

class Env:
    def __init__(self, level_index = 0, game_index = 0, dataset_index = 1, no_augment = True):
        # 初始化
        self.LEVELS = ['easy', 'medium', 'hard']
        self.DATASET_NAMES = ['train', 'test', 'valid']
        self.PATH_PREFIX = '/home/taku/research/zhuobinggang/commonsense-rl/games/twc'
        self.env = self.get_game_env(level_index, game_index, dataset_index)
        # act
        self.RIGHT_POSITION_HINT = 'Right position.'
        self.WRONG_POSITION_HINT = 'Wrong position, you should put it somewhere else, maybe the other room.'
        # 反馈强化
        self.no_augment = no_augment

    def initiate_env(self, env):
        if not hasattr(env, 'action_obs_pairs'):
            env.action_obs_pairs = []
        if not hasattr(env, 'system_user_msgs'): # 2024.5.22
            env.system_user_msgs = []
        if not hasattr(env, 'gpt_responses'): # 2024.5.22
            env.gpt_responses = []
        if not hasattr(env, 'available_actions'): # 2024.1.8 增加env.available_actions
            env.available_actions = []
        if not hasattr(env, 'last_reward'): # 2024.1.8 增加env.instant_reward
            env.last_reward = 0
            env.instant_reward = 0
        if not hasattr(env, 'score_by_step'): # 2024.2.9 增加env.score_by_step
            env.score_by_step = []
        if not hasattr(env, 'end'): # 2024.5.22
            env.end = False
        if not hasattr(env, 'info'): # 2024.5.23
            env.info = None # Will be injected in taku.step()
        if not hasattr(env, 'readable_log'): # 2024.8.9
            env.readable_log = '' 
    
    def get_game_name(self, level_index = 0, game_index = 0, dataset_index = 1):
        import glob
        level_name = self.LEVELS[level_index]
        dataset_name = self.DATASET_NAMES[dataset_index]
        GAME_NAMES = glob.glob(f'{self.PATH_PREFIX}/{level_name}/{dataset_name}/*ulx')
        game_name = GAME_NAMES[game_index]
        print(game_name)
        return game_name

    def get_game_env(self, level_index = 0, game_index = 0, dataset_index = 1):
        game_path = self.get_game_name(level_index, game_index, dataset_index)
        env = self.get_game_env_by_path(game_path)
        env.meta_info = f'get_game_env({level_index},{game_index},{dataset_index})'
        env.meta_name = f'{self.DATASET_NAMES[dataset_index]}_{self.LEVELS[level_index]}_{game_index}'
        self.initiate_env(env)
        return env

    def get_game_env_by_path(self, game_path, need_reset = True):
        from textworld import EnvInfos
        from games import dataset
        infos_to_request = EnvInfos(description=True, inventory=True, admissible_commands=True, won=True, lost=True,location = True,
                last_action=True, facts=True,entities=True, max_score = True, moves = True, score = True)
        env, game_file_names = dataset.get_game_env(game_path, infos_to_request, max_episode_steps = 50, mode = 'test')
        if need_reset:
            obs, infos = env.reset()
        return env

    ######################### 初始化END ###################

    def _step(self, cmd):
        obs, reward, is_not_terminal, infos = self.env.step([cmd])
        self.env.obs = obs # Added 2024.8.9
        self.env.info = infos # Added 2024.5.23
        return obs, infos

    def is_placing_item(self, command):
        command = command.lower()
        return command.split()[0] in ['put', 'insert']

    def act(self, command = None, no_augment = None):
        #initiate_env(env)
        env = self.env
        action_obs_pairs = env.action_obs_pairs
        no_augment = no_augment or self.no_augment
        if command:
            obs, info = self._step(command)
            # CLEAN OBS
            obs = obs[0].strip().replace('\n','')
            inventory = info['inventory'][0].strip().replace('\n','')
            description = info['description']
            description = description[0].strip().replace('\n','')
            # 如果obs和description重复，应该被截断
            if obs == description:
                obs = obs.split('.')[0] + '.'
            available_actions = info['admissible_commands'][0]
            # 记录瞬时奖励
            new_reward = info['score'][0]
            env.instant_reward = new_reward - env.last_reward
            env.last_reward = new_reward
            # 2024.1.21  反馈强化
            if no_augment:
                pass
            else:
                if env.instant_reward != 0:
                    obs += self.RIGHT_POSITION_HINT
                else:
                    if self.is_placing_item(command):
                        obs_lower = obs.lower()
                        if obs_lower.startswith("you can't") or obs_lower.startswith('you need'): # 放置失败
                            pass # 什么也不做
                        else:
                            obs += self.WRONG_POSITION_HINT # 放置成功，但是位置错误
            action_obs_pairs.append((command, obs))
        else: # 重新开始的情况
            print('RESTAR\n\n')
            _, info = env.reset()
            description = info['description']
            description = description[0].strip().replace('\n','')
            inventory = info['inventory'][0].strip().replace('\n','')
            available_actions = info['admissible_commands'][0]
            action_obs_pairs = []
            env.last_reward = 0
            env.instant_reward = 0
        env.available_actions = available_actions
        if info['won'][0]:
            env.end = True
            # 打引结束信息
            print(f"YOU WIN, score at {info['score']}/{info['max_score']}, steps {info['moves']}")
        return description, inventory, available_actions, action_obs_pairs


