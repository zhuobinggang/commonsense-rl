# BERT + reinforcement learning for FTWP dataset.
# 先这样：只保留房间名，行动历史，菜谱，命令列表，然后用4000多个游戏进行行为克隆
import common
from interface_ftwp import game_for_test, Ftwp_interface_by_path
from functools import lru_cache
from bert_common import Game_state, get_loss, get_optimizer, Abs_model_policy_gradient, initiate_bert, default_tokenizer

def game_for_test():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0()
    game = Game_policy_gradient(file_paths[1])
    game.verbose = True
    return game

class Prompt_builder_for_bert:
    def __init__(self):
        self.room_name = ''
        self.action_history = ''
        self.action_list = ''
        # others
        self.recipe = ''
        self.prompt = ''

    def build(self):
        user_msg = ''
        user_msg += f"Room: {self.room_name if self.room_name else 'Unknown now'}\n"
        user_msg += f"Recipe: {self.recipe if self.recipe else 'Unknown now'}\n"
        user_msg += f"Action history: {self.action_history if self.action_history else ''}\n" 
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += 'Next action: [MASK]'
        self.prompt = user_msg

def prompt_from_env_feedback_action_template(description, action_obs_pairs, available_actions, recipe):
    promptor = Prompt_builder_for_bert()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs, seperator='>')
    promptor.room_name = common.extract_room_name(description)
    promptor.action_list = common.actions_to_list_number(available_actions)
    promptor.recipe = recipe
    promptor.build()
    return promptor.prompt

def bert_prompt_from_game(game):
    return prompt_from_env_feedback_action_template(game.description, game.action_obs_pairs, game.available_actions, game.recipe)

def get_input_text(game):
    pass


class Game_for_bert(Ftwp_interface_by_path):
    def init_hook(self):
        pass
    def input(self, command):
        self.command = command
        try:
            x = bert_prompt_from_game(self)
            y = self.available_actions.index(command)
            self.finetune_triples.append((x, y)) # save x and y
            self.act_and_output(command)
        except:
            print(f'指令不存在，不需要执行: {command}')
            print(self.available_actions)
            print(self.filename)
    def save_as_json(self):
        f = open(f'exp/auto_filename/{self.filename}', 'w')
        for x, y in self.finetune_triples:
            obj = {'x': x.strip(), 'y': str(y).strip()}
            f.write(str(obj) + '\n')
        f.close()
    def output_actions(self):
        actions = [y for x, y in self.finetune_triples]
        return actions
    def save_readable(self):
        filename = self.filename + f'score{self.get_score()}_of_{self.get_max_score()}'
        f = open(f'exp/auto_filename/{filename}.txt', 'w')
        for x, y in self.finetune_triples:
            f.write(f'{x}\n\n{y}\n\n')
        f.write(f'{self.get_score()}/{self.get_max_score()}')
        f.close()
    def filter_walkthroughs(self, walkthroughs = None):
        walkthroughs = walkthroughs or self.get_walkthrough()
        self.verbose = False;
        self.reset()
        filtered_walkthroughs = []
        for action in walkthroughs:
            if action in self.filtered_commands:
                filtered_walkthroughs.append(action)
                self.input(action)
            else:
                print(f'filtered action: {action}')
        self.reset()
        return filtered_walkthroughs
    def auto_play(self, action_list):
        self.reset()
        for action in action_list:
            self.input(action)
    def get_x(self):
        return bert_prompt_from_game(self)
    def action_obs_pairs_got_callback(self, action_obs_pairs):
        self.save_if_checking_recipe(action_obs_pairs) 
        self.save_instant_reward(action_obs_pairs)
    def save_instant_reward(self, action_obs_pairs):
        self.instant_reward = 0
        if not action_obs_pairs or len(action_obs_pairs) < 1:
            return
        act, obs = action_obs_pairs[-1]
        if 'Your score has just gone up by one point' in obs:
            self.instant_reward = 1
    def get_instant_reward(self):
        return self.instant_reward

# DONE: TEST
def test_get_recipe():
    game = game_for_test()
    game.verbose = False
    game.reset()
    game.input('go west')
    game.input('go south')
    game.input('examine cookbook')
    # print(game.recipe)
    print(bert_prompt_from_game(game))
    return game


def game_played_and_save_training_file(game, output_file_path):
    import json
    game.verbose = False
    game.auto_play(game.get_walkthrough()) # TODO: 修改game游玩时生成的prompt
    f = open(output_file_path, 'w')
    for x, y in game.finetune_triples:
        obj = {'x': x.strip(), 'y': str(y).strip()}
        line = json.dumps(obj)
        f.write(line + '\n')
    f.close()

def batch_training_file_prepare():
    from ftwp_info import all_train_game_paths
    file_paths = all_train_game_paths() 
    bad_train_files = []
    for file_path in file_paths:
        game = Game_for_bert(file_path)
        game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')
        if not game.won:
            print(f'有问题啊朋友，文件数量: {len(bad_train_files)}')
            bad_train_files.append(file_path)
    return bad_train_files


def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare(directory_path = '/home/taku/Downloads/cog2019_ftwp/procceeded_training_files/bert_trains', out_path='/home/taku/Downloads/cog2019_ftwp/procceeded_training_files/bert_trains.jsonl')

# ==== auto play ========

def load_trained_model(path):
    from transformers import AutoTokenizer, ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBertForMaskedLM.from_pretrained(path)
    return model, tokenizer



@lru_cache(maxsize=None)
def command_indexs_tokenized(command_length = 100):
    tokenizer = default_tokenizer()
    command_index_string = ' '.join([str(item) for item in list(range(command_length))])
    return tokenizer.encode(command_index_string, add_special_tokens = False)


def construct_game_state(game: Game_for_bert):
    game_state = Game_state()
    game_state.x = game.get_x()
    game_state.action_list = game.filtered_commands
    return game_state

def trained_model_autoplay(game, model, tokenizer, save_readable = True):
    model.eval()
    game.reset()
    counter = 0
    while not any([counter >= 30, game.is_won(), game.is_lost()]):
        try:
            # command = get_next_command(game, tokenizer, model)
            # command = get_next_command_by_distribution(game, tokenizer, model) # NOTE: 2025.2.12 用sample取代原来的argmax
            # command = get_next_command_by_command_logits_argmax(game, tokenizer, model) # NOTE: 2025.2.12 用sample取代原来的argmax
            game_state = construct_game_state(game)
            command = get_next_command_by_command_logits_argmax_simple(model, game_state)
            game.input(command)
            counter += 1
        except Exception as ex:
            print(f'根据BERT获取指令出问题了，返回分数0即可')
            print(ex)
            return 0, 0
    if save_readable:
        game.save_readable()
    return game.get_score(), game.get_max_score()
    # logits = model(**inputs).logits
    # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

# 首先要获得x, 然后通过model获得logits输出，然后获得mask对应的logits
""" def get_mask_logits(game, tokenizer, model):
    import torch
    device = model.device
    x = game.get_x()
    inputs = tokenizer(x, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    return logits[0, mask_token_index] # (1, 50368) """

# TODO: 代换掉原来的函数
# @parameter: x是包含[MASK]标记的prompt
def get_mask_logits_simple(model, x):
    import torch
    device = model.device
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    return logits[0, mask_token_index] # (1, 50368)

# 用于critic
# @parameter: x是包含[MASK]标记的prompt
def get_cls_output(model, x):
    import torch
    device = model.device
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    out = model(**inputs.to(device), output_hidden_states=True) # 23 layers tuple
    last_layer = out.hidden_states[-1] # (1, 52, 768)
    cls_out = last_layer[:, 0] # (1, 768)
    return cls_out


# 需要从game处获得filtered_commands，那可以让他直接传入
""" def get_command_logits(game, tokenizer, model):
    mask_logits = get_mask_logits(game, tokenizer, model) # (1, 50368)
    command_length = len(game.filtered_commands)
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = mask_logits[0, command_indexs] # (command_length)
    return command_logits # (command_length)
 """
def get_command_logits_simple(model, state: Game_state):
    mask_logits = get_mask_logits_simple(model, state.x) # (1, 50368)
    command_length = len(state.action_list)
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = mask_logits[0, command_indexs] # (command_length)
    return command_logits # (command_length)

""" def get_command_distribution(game, tokenizer, model):
    command_logits = get_command_logits(game, tokenizer, model)
    command_logits[command_logits < 0] = 0 # 出现负数无法用于建构distribution，会报错，因此直接设置为0即可
    # print('=========')
    # print(command_logits) # NOTE: need test
    # print('=========')
    import torch
    dist = torch.distributions.categorical.Categorical(probs = command_logits)
    return dist """

def get_command_distribution_simple(model, state: Game_state):
    command_logits = get_command_logits_simple(model, state)
    print(command_logits) # NOTE: TESTING
    command_logits[command_logits < 0] = 0 # 出现负数无法用于建构distribution，会报错，因此直接设置为0即可
    import torch
    dist = torch.distributions.categorical.Categorical(probs = command_logits)
    return dist

def get_next_command(game, tokenizer, model):
    mask_logits = get_mask_logits(game, tokenizer, model)
    predicted_token_id = mask_logits.argmax(axis=-1) # TODO: 2025.1.20 为了能够对应强化学习，应该将argmax改为sampling
    command_str = tokenizer.decode(predicted_token_id).strip()
    try:
        command_index = int(command_str)
        command = game.filtered_commands[command_index]
    except Exception as ex:
        print(f'错误解析command: {command_str}\n in {game.available_actions}')
        print(ex)
        raise ex
    return command

# NOTE: Added 2025.2.12 理论上可以根绝错误输出的问题
""" def get_next_command_by_distribution(game, tokenizer, model):
    dist = get_command_distribution(game, tokenizer, model)
    command_index = dist.sample().item()
    command = game.filtered_commands[command_index]
    return command """

def get_next_command_by_distribution_simple(model, state: Game_state):
    dist = get_command_distribution_simple(model, state)
    command_index = dist.sample().item()
    command = state.action_list[command_index]
    return command

# Added 2025.2.12 限定版的argmax
""" def get_next_command_by_command_logits_argmax(game, tokenizer, model):
    command_logits = get_command_logits(game, tokenizer, model) # (command_length)
    command_index = command_logits.argmax().item()
    command = game.filtered_commands[command_index]
    return command """

# TODO: 测试
def get_next_command_by_command_logits_argmax_simple(model, state: Game_state):
    command_logits = get_command_logits_simple(model, state) # (command_length)
    command_index = command_logits.argmax().item()
    command = state.action_list[command_index]
    return command

# @history: 1.17: 需要分开彻底分开
def valid_paths():
    from ftwp_info import all_valid_game_paths
    return all_valid_game_paths(shuffle = True)[-30:]

def batch_test(model, tokenizer = None, save_readable = True, test_game_paths = [], file_prefix = ''):
    if not tokenizer:
        tokenizer = default_tokenizer()
    if len(test_game_paths) < 1:
        test_game_paths = valid_paths()
    scores = []
    max_scores = []
    for path in test_game_paths:
        game = Game_for_bert(path)
        game.verbose = False
        game.filename = file_prefix + game.filename # Added 2025.2.8
        score, max_score = trained_model_autoplay(game, model, tokenizer, save_readable)
        scores.append(score)
        max_scores.append(max_score)
    return sum(scores) / sum(max_scores)

def batch_valid(model, tokenizer = None, save_readable = True):
    return batch_test(model, tokenizer, save_readable, test_game_paths=valid_paths())
    
# ================= compare llm performance ===================

# @history: 2024.1.17
def run_test_temp(model):
    from ftwp_info import temp_test_valid_set
    test_game_paths, _ = temp_test_valid_set()
    return batch_test(model, save_readable=True, test_game_paths=test_game_paths)


def run_test(model):
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    return batch_test(model, save_readable=True, test_game_paths=test_game_paths)

def final_test():
    model_paths = ['/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart0.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart1.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart2.tch']
    results = [] # 3 models, 2 test methods
    for model_path in model_paths:
        model, toker = load_trained_model(model_path)
        model.cuda()
        # temp_score = run_test_temp(model)
        temp_score = 0
        real_score = run_test(model)
        results.append((temp_score, real_score))
    return results

def run_test_full(model, file_prefix = ''):
    from ftwp_info import all_test_game_paths
    test_game_paths=  all_test_game_paths()
    return batch_test(model, save_readable=True, test_game_paths=test_game_paths, file_prefix=file_prefix)

def run_test_full_with_model():
    import numpy as np
    model_paths = ['/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart0.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart1.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart2.tch']
    results = [] # 3 models, 2 test methods
    logger = common.Logger_simple(file_name='run_test_full_with_model_log')
    for model_idx, model_path in enumerate(model_paths):
        model, toker = load_trained_model(model_path)
        model.cuda()
        logger.add(f'Model {model_idx} testing...')
        score = run_test_full(model, file_prefix=f'M{model_idx}')
        logger.add(f'Model {model_idx} tested! Score = {score}.')
        logger.write_txt_log()
        results.append(score)
    logger.add(f'All model tested! Average score = {np.mean(results)}')
    logger.write_txt_log()
    return results


# ======================== 为policy gradient准备的 ==========================
# TODO: 测试
class Model_policy_gradient(Abs_model_policy_gradient):
    def __init__(self, bert) -> None:
        bert.cuda()
        self.bert = bert
        self.init_accelerate()
    def init_accelerate(self):
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        optimizer = get_optimizer(self.bert)
        bert, optimizer = accelerator.prepare(
            self.bert, optimizer
        )
        self.accelerator = accelerator
        self.bert = bert
        self.optimizer = optimizer
        self.device = device
    def clean_gradient(self):
        self.optimizer.zero_grad()
    def next_action(self, state: Game_state):
        return get_next_command_by_distribution_simple(self.bert, state)
    def update_policy(self, loss):
        # 需要观察之前的梯度下降是怎样和训练库联动的
        self.clean_gradient()
        self.accelerator.backward(loss)
        self.optimizer.step()
    def action_select_loss(self, state: Game_state, action, reward_scalar):
        # 直接用MLM损失来让模型最大化对某个token的输出
        try:
            action_idx = state.action_list.index(action) # 如果不存在，不需要计算loss
        except:
            return 0
        y = action_idx
        loss = get_loss(self.bert, state.x, y, device=self.device)
        # raw_loss = loss.item() # 在被放大之前输出
        loss = reward_scalar * loss
        return loss
    def eval(self):
        self.bert.eval()
    def train(self):
        self.bert.train()


# TODO: 测试
class Game_policy_gradient(Game_for_bert):
    def get_state(self):
        state = construct_game_state(self)
        return state
    def act(self, action):
        if action not in self.filtered_commands:
            print('???????????????????')
            print(action)
            print(self.filtered_commands)
            return 0
        self.input(action)
        reward = self.get_instant_reward()
        return reward
    
class Policy_gradient_tester:
    def __init__(self) -> None:
        self.model = Model_policy_gradient(initiate_bert())
        from ftwp_info import all_valid_game_paths
        self.game = Game_policy_gradient(all_valid_game_paths()[0])
        from common import Logger_simple
        self.txtLogger= Logger_simple('测试用logger')
        from bert_common import Logger_loss_and_score
        self.imageLogger = Logger_loss_and_score('图像用logger')
        self.counter = 99
        self.last_valid_score = 0
        self.valid_scores = []
        self.max_valid_score = 0

    def maybe_valid(self):
        self.counter += 1
        if self.counter % 100 == 0:
            self.last_valid_score = batch_valid(self.model.bert, None)
            if self.last_valid_score > self.max_valid_score:
                self.max_valid_score = self.last_valid_score
                self.save_checkpoint()
        self.valid_scores.append(self.last_valid_score)
        if self.counter % 100 == 0:
            self.imageLogger.rewards_per_episode = self.valid_scores
            self.imageLogger.losses_per_episode = [0] * len(self.valid_scores) # fake loss as 0
            self.imageLogger.episode_log()

    def save_checkpoint(self):
        checkpoint_path = f'exp/auto_filename/dd.tch'
        self.model.bert.save_pretrained(checkpoint_path)

    def test_policy_gradient(self):
        from bert_policy_tune import train_one_episode
        train_one_episode(self.model, self.game, txtLogger=self.txtLogger)

    def test_policy_gradient_200_episode(self):
        from bert_policy_tune import train_one_episode
        for i in range(200):
            train_one_episode(self.model, self.game, txtLogger=self.txtLogger, chartLogger=self.imageLogger)
        checkpoint_path = f'exp/auto_filename/dd.tch'
        self.model.bert.save_pretrained(checkpoint_path)

    def test_policy_gradient_with_walkthrough(self):
        from bert_policy_tune import train_one_episode
        train_one_episode(self.model, self.game, walkthrough=self.game.get_walkthrough(), txtLogger=self.txtLogger)

    def test_policy_gradient_200_episode_with_startup(self):
        from bert_policy_tune import train_one_episode
        # NOTE: 先用walkthrough初始化一下
        train_one_episode(self.model, self.game, walkthrough=self.game.get_walkthrough(), txtLogger=self.txtLogger, chartLogger=self.imageLogger)
        for i in range(200):
            train_one_episode(self.model, self.game, txtLogger=self.txtLogger, chartLogger=self.imageLogger)
        checkpoint_path = f'exp/auto_filename/dd.tch'
        self.model.bert.save_pretrained(checkpoint_path)
    
    def train_explore_1v1(self):
        from bert_policy_tune import train_one_episode
        from ftwp_info import all_train_game_paths
        paths = all_train_game_paths()
        game_count = len(paths)
        for game_index, path in enumerate(paths):
            self.txtLogger.add(f'训练中: {game_index}/{game_count}')
            game = Game_policy_gradient(path)
            # 探索
            train_one_episode(self.model, game, txtLogger=self.txtLogger, chartLogger=None)
            # 监督学习，不需要任何记录
            train_one_episode(self.model, game, walkthrough=game.get_walkthrough()) 
            self.maybe_valid() # 同时保存检查点
        self.imageLogger.episode_log()

    def train_only(self):
        # 区别仅在于每一步训练，或是整个游戏训练
        from bert_policy_tune import train_one_episode
        from ftwp_info import all_train_game_paths
        paths = all_train_game_paths()
        game_count = len(paths)
        for game_index, path in enumerate(paths):
            self.txtLogger.add(f'训练中: {game_index}/{game_count}')
            game = Game_policy_gradient(path)
            # 监督学习，不需要任何记录
            train_one_episode(self.model, game, walkthrough=game.get_walkthrough()) 
            self.maybe_valid() # 同时保存检查点
        self.imageLogger.episode_log()

    def train_explore_1v1_test(self):
        from bert_policy_tune import train_one_episode
        for _ in range(200):
            game = self.game
            # 探索
            train_one_episode(self.model, game, txtLogger=self.txtLogger, chartLogger=self.imageLogger)
            # 监督学习，不需要任何记录
            train_one_episode(self.model, game, walkthrough=game.get_walkthrough()) 
        self.save_checkpoint()

    def train_just_walkthrough_test(self):
        from bert_policy_tune import train_one_episode
        from ftwp_info import all_train_game_paths
        game = self.game
        counter = 0
        for _ in range(200):
            counter += 1
            if counter % 10 == 0:
                score, max_score = trained_model_autoplay(self.game, self.model.bert, None)
                norm_score = score / max_score
                self.txtLogger.add(f'测试结果: {norm_score}')
            train_one_episode(self.model, game, walkthrough=game.get_walkthrough(), txtLogger=self.txtLogger, chartLogger=self.imageLogger)

