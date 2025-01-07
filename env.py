# 重写一切


class Env:

    def __init__(self,
                 level_index=0,
                 game_index=0,
                 dataset_index=1,
                 no_augment=True):
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
        if not hasattr(env, 'system_user_msgs'):  # 2024.5.22
            env.system_user_msgs = []
        if not hasattr(env, 'gpt_responses'):  # 2024.5.22
            env.gpt_responses = []
        if not hasattr(
                env, 'available_actions'):  # 2024.1.8 增加env.available_actions
            env.available_actions = []
        if not hasattr(env, 'last_reward'):  # 2024.1.8 增加env.instant_reward
            env.last_reward = 0
            env.instant_reward = 0
        if not hasattr(env, 'score_by_step'):  # 2024.2.9 增加env.score_by_step
            env.score_by_step = []
        if not hasattr(env, 'end'):  # 2024.5.22
            env.end = False
        if not hasattr(env, 'info'):  # 2024.5.23
            env.info = None  # Will be injected in taku.step()
        if not hasattr(env, 'readable_log'):  # 2024.8.9
            env.readable_log = ''

    def get_game_name(self, level_index=0, game_index=0, dataset_index=1):
        import glob
        level_name = self.LEVELS[level_index]
        dataset_name = self.DATASET_NAMES[dataset_index]
        GAME_NAMES = glob.glob(
            f'{self.PATH_PREFIX}/{level_name}/{dataset_name}/*ulx')
        game_name = GAME_NAMES[game_index]
        print(game_name)
        return game_name

    def get_game_env(self, level_index=0, game_index=0, dataset_index=1):
        game_path = self.get_game_name(level_index, game_index, dataset_index)
        env = self.get_game_env_by_path(game_path)
        env.meta_info = f'get_game_env({level_index},{game_index},{dataset_index})'
        env.meta_name = f'{self.DATASET_NAMES[dataset_index]}_{self.LEVELS[level_index]}_{game_index}'
        self.initiate_env(env)
        return env

    def get_game_env_by_path(self, game_path, need_reset=True):
        from textworld import EnvInfos
        from games import dataset
        infos_to_request = EnvInfos(description=True,
                                    inventory=True,
                                    admissible_commands=True,
                                    won=True,
                                    lost=True,
                                    location=True,
                                    last_action=True,
                                    facts=True,
                                    entities=True,
                                    max_score=True,
                                    moves=True,
                                    score=True)
        env, game_file_names = dataset.get_game_env(game_path,
                                                    infos_to_request,
                                                    max_episode_steps=50,
                                                    mode='test')
        if need_reset:
            obs, infos = env.reset()
        return env

    ######################### 初始化END ###################

    def get_moves(self, info):
        return info['moves'][0]
    
    def origin_env_step(self, cmd):
        return self.env.step([cmd])

    def _step(self, cmd):
        # Add 2024.8.17 需要判断行动是否成功，如果不成功返回None
        # @history 2025.1.7 考虑到一些命令，比如检查库存，以及开门等等，不会增加步数，且错误的指令也会得到反馈，因此建议不改编引擎行为，返回全部，并且增加额外判断信息。
        old_moves = self.get_moves(self.env.info)
        obs, reward, is_not_terminal, info = self.origin_env_step(cmd)
        new_moves = self.get_moves(info)
        info['taku_info'] = {}
        if old_moves == new_moves:  # Add 2024.8.17 需要判断行动是否成功，如果不成功返回None
            info['taku_info'] = {'status': 2, 'desc': '步数不增加，需要小心'}
            print(f'TAKU: env._step WARNING, {cmd} 步数不增加，需要小心!但是返回环境反馈')
            self.err = {'obs': obs, 'info': info}
            self.env.err = {'obs': obs, 'info': info}
            return obs, info
        elif new_moves == old_moves + 1:  # SUCCESS
            info['taku_info'] = {'status': 1, 'desc': '成功改变环境状态'}
            self.env.obs = obs  # Added 2024.8.9
            self.env.info = info  # Added 2024.5.23
            return obs, info
        else:
            info['taku_info'] = {'status': 3, 'desc': '不应该进入的分支'}
            print('TAKU: ERRRRRRRRRRRRRRRRRRR IMPOSIBLE!')
            return None, None

    def is_placing_item(self, command):
        command = command.lower()
        return command.split()[0] in ['put', 'insert']
    
    def get_action_obs_pair(self):
        return self.env.action_obs_pairs

    def append_command_obs_pair(self, command, obs):
        self.env.action_obs_pairs.append((command, obs))

    def is_won(self, info):
        return info['won'][0]
    
    def is_lost(self, info):
        return info['lost'][0]

    def get_desc(self, info):
        return info['description'][0].strip().replace('\n', '')
    
    def get_inventory(self, info):
        return info['inventory'][0].strip().replace('\n', '')
    
    def get_available_actions(self, info):
        return info['admissible_commands'][0]
    
    def get_score(self, info):
        return info['score'][0]
    
    def get_obs(self, obs_raw):
        return obs_raw[0].strip().replace('\n', '')
    
    def reset(self, env):
        _, info = env.reset()
        info['taku_info'] = {}
        return _, info

    def act(self, command=None, no_augment=None):
        #initiate_env(env)
        env = self.env
        no_augment = no_augment or self.no_augment
        if command:
            # Add 2024.8.17 需要判断行动是否成功，如果不成功则直接跳出
            obs_raw, info = self._step(command)
            # CLEAN OBSERVATION
            obs = self.get_obs(obs_raw)
            inventory = self.get_inventory(info)
            description = self.get_desc(info)
            if obs.replace(' ', '') == description.replace(' ', ''):  # 如果obs和description重复，应该被截断以保证输入简洁
                obs = obs.split('.')[0] + '.'
            # Available actions
            env.available_actions = self.get_available_actions(info)
            # 记录瞬时奖励
            new_reward = self.get_score(info)
            env.instant_reward = new_reward - env.last_reward
            env.last_reward = new_reward
            # 2024.1.21  反馈强化
            if not no_augment:
                if env.instant_reward != 0:
                    obs += self.RIGHT_POSITION_HINT
                    print(f'FA: {obs}')
                else:
                    if self.is_placing_item(command):
                        obs_lower = obs.lower()
                        if obs_lower.startswith(
                                "you can't") or obs_lower.startswith(
                                    'you need'):  # 放置失败
                            pass  # 什么也不做
                        else:
                            obs += self.WRONG_POSITION_HINT  # 放置成功，但是位置错误
                            print(f'FA: {obs}')
            # 记录历史
            self.append_command_obs_pair(command, obs);
        else:  # 重新开始的情况
            print('RESTAR\n\n')
            _, info = self.reset(env)
            env.info = info  # Add 2024.8.17
            description = self.get_desc(info)
            inventory = self.get_inventory(info)
            env.available_actions = self.get_available_actions(info)
            env.action_obs_pairs = []
            env.last_reward = 0
            env.instant_reward = 0
        if self.is_won(info):
            env.end = True
            # 打引结束信息
            print(
                f"YOU WIN, score at {info['score']}/{info['max_score']}, steps {info['moves']}"
            )
            return None, None, None, None
        return description, inventory, env.available_actions, env.action_obs_pairs


class Env_extra_info(Env):
    def act(self, command=None, no_augment=None):
        #initiate_env(env)
        env = self.env
        no_augment = no_augment or self.no_augment
        if command:
            # Add 2024.8.17 需要判断行动是否成功，如果不成功则直接跳出
            obs_raw, info = self._step(command)
            # CLEAN OBSERVATION
            obs = self.get_obs(obs_raw)
            inventory = self.get_inventory(info)
            description = self.get_desc(info)
            if obs.replace(' ', '') == description.replace(' ', ''):  # 如果obs和description重复，应该被截断以保证输入简洁
                obs = obs.split('.')[0] + '.'
                info['obs_abbreviation'] = True
            # Available actions
            env.available_actions = self.get_available_actions(info)
            # 记录瞬时奖励
            new_reward = self.get_score(info)
            env.instant_reward = new_reward - env.last_reward
            env.last_reward = new_reward
            # 2024.1.21  反馈强化
            if not no_augment:
                if self.is_placing_item(command):
                    info['is_placing_item'] = True
                    if env.instant_reward != 0:
                        obs += self.RIGHT_POSITION_HINT
                        info['placing_failed'] = False
                        info['point_up'] = True
                        # print(f'FA: {obs}')
                    else:
                        info['placing_failed'] = True
                        obs_lower = obs.lower()
                        if obs_lower.startswith(
                                "you can't") or obs_lower.startswith(
                                    'you need'):  # 放置失败
                            pass  # 什么也不做
                        else:
                            obs += self.WRONG_POSITION_HINT  # 放置成功，但是位置错误
                            # print(f'FA: {obs}')
            # 记录历史
            self.append_command_obs_pair(command, obs);
        else:  # 重新开始的情况
            print('RESTAR\n\n')
            _, info = self.reset(env)
            info['taku_info']['desc'] = '重新启动游戏'
            env.info = info  # Add 2024.8.17
            description = self.get_desc(info)
            inventory = self.get_inventory(info)
            env.available_actions = self.get_available_actions(info)
            env.action_obs_pairs = []
            env.last_reward = 0
            env.instant_reward = 0
            # Add 2024.12.16
            env.max_score = info['max_score']
        self.info = info
        won = self.is_won(info)
        lost = self.is_lost(info)
        if lost or won:
            env.end = True
            print(f"YOU WIN, score at {info['score']}/{info['max_score']}, steps {info['moves']}") if won else print(f"YOU LOST, score at {info['score']}/{info['max_score']}, steps {info['moves']}")
            return None, None, None, None
        return description, inventory, env.available_actions, env.action_obs_pairs