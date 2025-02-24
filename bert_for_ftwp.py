# BERT + reinforcement learning for FTWP dataset.
# 先这样：只保留房间名，行动历史，菜谱，命令列表，然后用4000多个游戏进行行为克隆
import common
from common import draw_line_chart
from bert_common import Game_state, get_loss, get_optimizer, Abs_model_policy_gradient, initiate_bert, default_tokenizer
from bert_common import bert_prompt_from_game, Game_for_bert, get_next_command_by_command_logits_argmax_simple
from bert_common import get_next_command_by_distribution_simple, load_trained_model, construct_game_state, Game_for_rl
from bert_common import trained_model_autoplay, batch_test, batch_valid
from interface_ftwp import game_for_test

def game_for_test():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0()
    game = Game_for_rl(file_paths[1])
    game.verbose = True
    return game


def get_input_text(game):
    pass

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
    def backward_loss(self, loss):
        self.accelerator.backward(loss)
    def update_policy(self, loss):
        # 需要观察之前的梯度下降是怎样和训练库联动的
        self.clean_gradient()
        self.backward_loss(loss)
        self.optimizer_step()
    def optimizer_step(self):
        self.optimizer.step()
    def update_policy_with_multiple_losses(self, losses):
        # 需要观察之前的梯度下降是怎样和训练库联动的
        self.clean_gradient()
        for loss in losses:
            self.accelerator.backward(loss)
        self.optimizer.step()
    def action_select_loss(self, state: Game_state, action, reward_scalar):
        # 直接用MLM损失来让模型最大化对某个token的输出
        action_idx = state.action_list.index(action) # 如果不存在，不需要计算loss
        y = action_idx
        loss = get_loss(self.bert, state.x, y, device=self.device)
        # raw_loss = loss.item() # 在被放大之前输出
        loss = reward_scalar * loss
        return loss
    def eval(self):
        self.bert.eval()
    def train(self):
        self.bert.train()


    
class Policy_gradient_tester:
    def __init__(self) -> None:
        self.model = Model_policy_gradient(initiate_bert())
        from ftwp_info import all_valid_game_paths
        self.game = Game_for_rl(all_valid_game_paths()[0])
        from common import Logger_simple
        self.txtLogger= Logger_simple('测试用logger')
        from bert_common import Logger_loss_and_score
        self.imageLogger = Logger_loss_and_score('图像用logger')
        self.counter = 99
        self.last_valid_score = 0
        self.valid_scores = []
        self.max_valid_score = 0
        self.init_params_for_log() # 2025.2.24

    def init_params_for_log(self):
        # for valid
        self.counter = 0;
        self.last_valid_score = 0;
        self.max_valid_score = 0;
        self.valid_scores = [];
        self.episode_rewards = [];
        self.actor_losses = [];
        self.critic_losses = [];

    def draw_line_chart(self):
        draw_line_chart(list(range(len(self.episode_rewards))), [self.valid_scores, self.actor_losses], ['r', 'a'])
    def maybe_valid_and_save(self):
        self.counter += 1
        need_valid_and_update = self.counter % 100 == 0
        if need_valid_and_update: # valid然后更新max_valid_score和last_valid_score，如果分数更大，保存checkpoint
            self.last_valid_score = batch_valid(self.model.bert, save_readable = False)
            if self.last_valid_score > self.max_valid_score:
                self.max_valid_score = self.last_valid_score
                self.save_checkpoint()
        self.valid_scores.append(self.last_valid_score)
        if need_valid_and_update: # 绘制中间图像
            self.draw_line_chart()

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
            game = Game_for_rl(path)
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
            game = Game_for_rl(path)
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
                score, max_score = trained_model_autoplay(self.game, self.model.bert)
                norm_score = score / max_score
                self.txtLogger.add(f'测试结果: {norm_score}')
            train_one_episode(self.model, game, walkthrough=game.get_walkthrough(), txtLogger=self.txtLogger, chartLogger=self.imageLogger)

    def train_only_20250224(self):
        # 区别仅在于每一步训练，或是整个游戏训练
        from bert_policy_tune import train_one_episode
        from ftwp_info import all_train_game_paths
        paths = all_train_game_paths()
        game_count = len(paths)
        for game_index, path in enumerate(paths):
            print(f'{game_index}/{game_count}')
            game = Game_for_rl(path)
            # self.txtLogger.add(f'训练中: {game_index}/{game_count}')
            # 监督学习，不需要任何记录
            norm_score, mean_actor_loss = train_one_episode(self.model, game, walkthrough=game.filter_walkthroughs_with_input_test()) 
            self.episode_rewards.append(norm_score)
            self.actor_losses.append(mean_actor_loss)
            self.maybe_valid_and_save() # 同时保存检查点
        self.draw_line_chart()