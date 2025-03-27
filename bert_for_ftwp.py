# BERT + reinforcement learning for FTWP dataset.
# 先这样：只保留房间名，行动历史，菜谱，命令列表，然后用4000多个游戏进行行为克隆
import common
from common import draw_line_chart
from bert_common import Game_state, get_loss, get_optimizer, Abs_model_policy_gradient, initiate_bert, default_tokenizer
from bert_common import bert_prompt_from_game, Game_for_bert, get_next_command_by_command_logits_argmax_simple
from bert_common import get_next_command_by_distribution_simple, load_trained_model, Game_for_rl
from bert_common import batch_test, batch_valid
from bert_common import run_test_full, first_train_game
from interface_ftwp import game_for_test
import torch
from torch import nn
from torch import optim

def game_for_test():
    from ftwp_info import all_game_paths
    file_paths = all_game_paths()
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
    return batch_test(model, test_game_paths=test_game_paths)


def run_test(model):
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    return batch_test(model, test_game_paths=test_game_paths)

def final_test():
    model_paths = ['/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart0.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart1.tch', 
              '/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart2.tch']
    results = [] # 3 models, 2 test methods
    for model_path in model_paths:
        model, toker = load_trained_model(model_path)
        model.cuda()
        temp_score = run_test_temp(model)
        temp_score = 0
        real_score = run_test(model)
        results.append((temp_score, real_score))
    return results

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

def run_test_policy_gradient_20250225():
    model, _ = load_trained_model('exp/auto_filename/dd.tch')
    model.cuda()
    temp_score = run_test_temp(model)
    real_score = run_test(model)
    return temp_score, real_score

# ======================== 为policy gradient准备的 ==========================
# TODO: 测试
class Model_policy_gradient(Abs_model_policy_gradient):
    def __init__(self, bert = None) -> None:
        if not bert:
            bert = initiate_bert()
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
    def init_optimizer(self):
        self.init_accelerate()
    def clean_gradient(self):
        self.optimizer.zero_grad()
    def next_action_with_explore(self, state: Game_state):
        return get_next_command_by_distribution_simple(self.bert, state)
    def next_action(self, state: Game_state):
        return get_next_command_by_command_logits_argmax_simple(self.bert, state)
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

# 简单的神经网络基线，chatgpt生成的，需要用BERT来重新实现

class SimpleBaseline(nn.Module):
    def __init__(self, state_dim, lr=1e-3):
        super(SimpleBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),  # 隐藏层 1
            nn.ReLU(),
            nn.Linear(32, 16),  # 隐藏层 2
            nn.ReLU(),
            nn.Linear(16, 1)  # 输出层 (V(s))
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # 使用均方误差损失函数

    def forward(self, state):
        if isinstance(state, list) or isinstance(state, tuple):
            state = torch.tensor(state, dtype=torch.float32)
        return self.net(state).squeeze(-1)  # 输出 V(s)，去掉不必要的维度

    def get_value(self, state):
        """返回状态值 V(s)，但不进行梯度计算"""
        with torch.no_grad():
            return self.forward(state).item()

    def update(self, state, G):
        """用 G 来更新 V(s)"""
        self.optimizer.zero_grad()
        pred_value = self.forward(state)  # 预测 V(s)
        target = torch.tensor(G, dtype=torch.float32)  # 目标 G
        loss = self.loss_fn(pred_value, target)  # 计算 MSE 损失
        loss.backward()  # 反向传播
        self.optimizer.step()  # 梯度更新

    
class Policy_gradient_tester:
    def __init__(self, trained_bert = None, save_prefix = 'policy_gradient') -> None:
        if trained_bert:
            self.actor = Model_policy_gradient(trained_bert)
            self.trained_bert_provided = True
        else:
            # self.actor = Model_policy_gradient(initiate_lora_bert())
            self.actor = Model_policy_gradient(initiate_bert())
            self.trained_bert_provided = False
        self.game = first_train_game()
        from common import Logger_simple
        self.txtLogger= Logger_simple(save_prefix)
        from bert_common import Logger_loss_and_score
        self.imageLogger = Logger_loss_and_score(save_prefix)
        self.counter = 99
        self.last_valid_score = 0
        self.valid_scores = []
        self.max_valid_score = 0
        self.init_params_for_log() # 2025.2.24
        self.save_prefix = save_prefix # 0303

    def init_params_for_log(self):
        # for valid
        self.counter = 0;
        self.last_valid_score = 0;
        self.max_valid_score = 0;
        self.valid_scores = [];
        self.episode_rewards = [];
        self.actor_losses = [];
        self.critic_losses = []
        self.log_steps = 100;

    def draw_line_chart(self):
        draw_line_chart(list(range(len(self.episode_rewards))), [self.valid_scores, self.actor_losses], ['valid_score', 'a_loss'], path = f'exp/auto_filename/{self.save_prefix}.png')

    def maybe_valid_and_save(self):
        self.counter += 1
        need_valid_and_update = self.counter % self.log_steps == 0
        if need_valid_and_update: # valid然后更新max_valid_score和last_valid_score，如果分数更大，保存checkpoint
            self.txtLogger.add(f'Valid中……')
            self.last_valid_score = batch_valid(self.actor.bert, save_readable = False)
            self.txtLogger.add(f'Valid分数为{self.last_valid_score}')
            self.txtLogger.write_txt_log()
            if self.last_valid_score > self.max_valid_score:
                self.max_valid_score = self.last_valid_score
                self.save_checkpoint()
        self.valid_scores.append(self.last_valid_score)
        if need_valid_and_update: # 绘制中间图像
            pass

    def save_checkpoint(self):
        checkpoint_path = f'exp/auto_filename/{self.save_prefix}.tch'
        self.actor.bert.save_pretrained(checkpoint_path)

    def test_policy_gradient(self):
        from policy_algorithms import train_one_episode
        train_one_episode(self.actor, self.game, txtLogger=self.txtLogger)

    def train_only_20250303(self):
        # 区别仅在于每一步训练，或是整个游戏训练
        self.counter = -1
        from policy_algorithms import train_one_episode
        from ftwp_info import all_train_game_paths
        paths = all_train_game_paths()
        game_count = len(paths)
        self.txtLogger.add('一个epoch训练开始！')
        for game_index, path in enumerate(paths):
            log_txt = f'{game_index}/{game_count}'
            print(log_txt)
            self.txtLogger.add(log_txt)
            game = Game_for_rl(path)
            # self.txtLogger.add(f'训练中: {game_index}/{game_count}')
            # 监督学习，不需要任何记录
            norm_score, mean_actor_loss = train_one_episode(self.actor, game, walkthrough=game.filter_walkthroughs_with_input_test()) 
            self.episode_rewards.append(norm_score)
            self.actor_losses.append(min(mean_actor_loss * 0.1, 1.5)) # NOTE: scale for image ouput
            self.maybe_valid_and_save() # 同时保存检查点
            if self.counter % self.log_steps == 0:
                self.draw_line_chart()
        self.draw_line_chart()
        self.txtLogger.add('一个epoch训练结束。')
        self.txtLogger.write_txt_log()

    
    def train_only_3_epoch_20250303(self):
        for e in range(3):
            self.train_only_20250303()

    def test_fixed_algorithm_explore_only(self):
        from policy_algorithms import train_one_episode
        for index in range(200):
            game = self.game
            norm_score, mean_actor_loss = train_one_episode(self.actor, game) 
            self.episode_rewards.append(norm_score)
            self.valid_scores = self.episode_rewards # use the same score to log
            self.actor_losses.append(mean_actor_loss)
            if index % 10 == 0:
                self.draw_line_chart()
        self.draw_line_chart()

    def train_explore1v1_20250224(self):
        # 更新算法之后重新实验
        from policy_algorithms import train_one_episode
        from ftwp_info import all_train_game_paths
        paths = all_train_game_paths()
        game_count = len(paths)
        for game_index, path in enumerate(paths):
            print(f'{game_index}/{game_count}')
            game = Game_for_rl(path)
            # 探索
            explore_norm_score, mean_actor_loss = train_one_episode(self.actor, game)
            # 监督学习，不需要任何记录
            _, _ = train_one_episode(self.actor, game, walkthrough=game.filter_walkthroughs_with_input_test()) 
            self.episode_rewards.append(explore_norm_score)
            self.actor_losses.append(mean_actor_loss * 0.1)
            self.maybe_valid_and_save() # 同时保存检查点
            if self.counter % self.log_steps == 0:
                draw_line_chart(list(range(len(self.episode_rewards))), [self.episode_rewards, self.valid_scores, self.actor_losses], ['explore_score', 'valid_score', 'a_loss'])
        draw_line_chart(list(range(len(self.episode_rewards))), [self.episode_rewards, self.valid_scores, self.actor_losses], ['explore_score', 'valid_score', 'a_loss'])

    def explore_augment(self):
        if not self.trained_bert_provided:
            raise Exception('该探索强化必须对于已经训练的模型使用,对于完全没有训练的模型基本上不会有效果!除非实装了baseline!')
        # 更新算法之后重新实验
        self.counter = -1 # 用于在一开始进行打印
        from policy_algorithms import train_one_episode
        from ftwp_info import all_train_game_paths
        paths = all_train_game_paths()
        game_count = len(paths)
        for game_index, path in enumerate(paths):
            print(f'{game_index}/{game_count}')
            game = Game_for_rl(path)
            # 探索
            explore_norm_score, mean_actor_loss = train_one_episode(self.actor, game)
            self.episode_rewards.append(explore_norm_score)
            self.actor_losses.append(mean_actor_loss * 0.1)
            self.maybe_valid_and_save() # 同时保存检查点
            if self.counter % self.log_steps == 0:
                draw_line_chart(list(range(len(self.episode_rewards))), [self.episode_rewards, self.valid_scores], ['explore_score', 'valid_score'])
        draw_line_chart(list(range(len(self.episode_rewards))), [self.episode_rewards, self.valid_scores], ['explore_score', 'valid_score'])


# ==== 2025.2.28 计算厨房访问对分数的影响 ====

def test_kitchen_visit_rate():
    from bert_common import run_valid_full
    model, _ = load_trained_model('/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart0.tch')
    model.cuda()
    run_valid_full(model, file_prefix='behavior_clone')
    model, _ = load_trained_model('/home/taku/Downloads/cog2019_ftwp/trained_models/policy_gradient/policy_gradient20250225.tch')
    model.cuda()
    run_valid_full(model, file_prefix='REINFORCE')

def batch_train_pure_REINFORCE():
    for i in range(3):
        tester = Policy_gradient_tester(save_prefix=f'pure_REINFORCE_{i}')
        tester.train_only_3_epoch_20250303()

def batch_test():
    for i in range(3):
        model, _ = load_trained_model(f'exp/auto_filename/pure_REINFORCE_{i}.tch')
        model.cuda()
        run_test_full(model, file_prefix=f'pure_REINFORCE_{i}')

def batch_test_behavior_clone():
    for i in range(3):
        model, _ = load_trained_model(f'/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0121/baseline_restart{i}.tch')
        model.cuda()
        run_test_full(model, file_prefix=f'behavior_clone_{i}')
    import os
    os.system('shutdown')