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
        old_moves = self.get_moves(self.env.info)
        obs, reward, is_not_terminal, info = self.origin_env_step(cmd)
        new_moves = self.get_moves(info)
        if old_moves == new_moves:  # Add 2024.8.17 需要判断行动是否成功，如果不成功返回None
            print(f'TAKU: env._step WARNING, {cmd} NOT EXECUTABLE!')
            self.err = {'obs': obs, 'info': info}
            self.env.err = {'obs': obs, 'info': info}
            return None, None
        elif new_moves == old_moves + 1:  # SUCCESS
            self.env.obs = obs  # Added 2024.8.9
            self.env.info = info  # Added 2024.5.23
            return obs, info
        else:
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

    def act(self, command=None, no_augment=None):
        #initiate_env(env)
        env = self.env
        no_augment = no_augment or self.no_augment
        if command:
            # Add 2024.8.17 需要判断行动是否成功，如果不成功则直接跳出
            obs_raw, info = self._step(command)
            if obs_raw is None:
                print(f'TAKU: env.act WARNING, {command} NOT EXECUTABLE!')
                return None, None, None, None  # 在该情况下，唯一被设置的东西是env.err
            # CLEAN OBSERVATION
            obs = self.get_obs(obs_raw)
            inventory = self.get_inventory(info)
            description = self.get_desc(info)
            if obs == description:  # 如果obs和description重复，应该被截断以保证输入简洁
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
            _, info = env.reset()
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

    def _step(self, cmd):
        # Add 2024.11.20 增加更多来自环境的反馈
        extra_info_dic = {}
        # Add 2024.8.17 需要判断行动是否成功，如果不成功返回None
        old_moves = self.get_moves(self.env.info)
        obs, reward, is_not_terminal, info = self.origin_env_step(cmd)
        new_moves = self.get_moves(info)
        if old_moves == new_moves:  # Add 2024.8.17 命令执行失败，obs被存储到env.err['obs']
            extra_info_dic = {'desc': f'步数没有增加，{cmd}执行失败，raw obs为{obs}', 'raw_obs': obs, 'env_act_failed': True}
            print(f'TAKU: env._step WARNING, {cmd} NOT EXECUTABLE!')
            self.err = {'obs': obs, 'info': info}
            self.env.err = {'obs': obs, 'info': info}
            return None, None, extra_info_dic
        elif new_moves == old_moves + 1:  # SUCCESS
            extra_info_dic = {'desc': f'步数增加，{cmd}执行成功，raw obs为{obs}', 'raw_obs': obs, 'env_act_failed': False}
            self.env.obs = obs  # Added 2024.8.9
            self.env.info = info  # Added 2024.5.23
            return obs, info, extra_info_dic
        else:
            print('TAKU: ERRRRRRRRRRRRRRRRRRR IMPOSIBLE!')
            extra_info_dic = {'desc': f'进入了不可能的领域', 'raw_obs': 'Unknown failure.', 'env_act_failed': True}
            return None, None, extra_info_dic
        
    def act(self, command=None, no_augment=None):
        #initiate_env(env)
        env = self.env
        no_augment = no_augment or self.no_augment
        extra_info = {}
        extra_info['env_act_failed'] = False
        extra_info['is_placing_item'] = False
        extra_info['placing_failed'] = False
        extra_info['won'] = False
        if command:
            # Add 2024.8.17 需要判断行动是否成功，如果不成功则直接跳出
            obs_raw, info, extra_info_raw = self._step(command)
            extra_info = extra_info | extra_info_raw # merge extra info
            if obs_raw is None: # 执行失败的情况，直接跳出
                print(f'TAKU: env.act WARNING, {command} NOT EXECUTABLE!')
                return None, None, None, None, extra_info # 2024.11.20 extra info
            # CLEAN OBSERVATION
            obs = self.get_obs(obs_raw)
            inventory = self.get_inventory(info)
            description = self.get_desc(info)
            if obs == description:  # 如果obs和description重复，应该被截断以保证输入简洁
                obs = obs.split('.')[0] + '.'
                extra_info['obs_abbreviation'] = True
            # Available actions
            env.available_actions = self.get_available_actions(info)
            # 记录瞬时奖励
            new_reward = self.get_score(info)
            env.instant_reward = new_reward - env.last_reward
            env.last_reward = new_reward
            # 2024.1.21  反馈强化
            if not no_augment:
                if self.is_placing_item(command):
                    extra_info['is_placing_item'] = True
                    if env.instant_reward != 0:
                        obs += self.RIGHT_POSITION_HINT
                        extra_info['placing_failed'] = False
                        extra_info['point_up'] = True
                        # print(f'FA: {obs}')
                    else:
                        extra_info['placing_failed'] = True
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
            _, info = env.reset()
            extra_info['desc'] = '重新启动游戏'
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
            extra_info['won'] = True
            return None, None, None, None, extra_info
        return description, inventory, env.available_actions, env.action_obs_pairs, extra_info