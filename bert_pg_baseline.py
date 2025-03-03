# Baseline implementation for BERT-based REINFORCE(policy gradient)

import torch
import torch.nn as nn
import torch.optim as optim
from bert_qac import Critic, encode_get_cls, squared_loss
import bert_qac
from bert_common import Abs_model_policy_gradient, initiate_bert, batch_valid
from common import draw_line_chart, Logger_simple
from bert_policy_tune import Fake_text_logger
from bert_for_ftwp import Model_policy_gradient

class SimpleBaseline(Critic):

    def forward(self, state):
        x = state.x
        cls_token = encode_get_cls(self.bert, x) # (1, 768)
        expect_score = self.mlp_scorer.expect_score(cls_token) # (1, 1) 类似于 [0.0784]
        return expect_score

    def get_value(self, state):
        """返回状态值 V(s)，但不进行梯度计算"""
        with torch.no_grad():
            return self.forward(state).item()

    def update(self, loss):
        return self.update_critic(loss)
    

# TODO: 测试
# TODO: 爆改logger记录每一步的损失值？最好能够记录这次优化的是哪个action，并且记录对应的损失值，瞬时reward也记录一下
def train_one_episode(model: Abs_model_policy_gradient, baseline, game, walkthrough = None, txtLogger = None):
    if not txtLogger:
        txtLogger = Fake_text_logger('')
    # 先使用walkthrough训练一次，然后自主探索3次
    episode_sars = [] # state, action, reward
    game.reset()
    counter = 0
    final_reward = 0
    model.eval()
    txtLogger.add('===== 探索中 =====')
    while not any([counter >= 50, game.is_won(), game.is_lost()]):
        state = game.get_state()
        if walkthrough:
            if counter < len(walkthrough):
                action = walkthrough[counter]
            else:
                print('到头了，但是游戏没有结束，说明walkthrough可能有问题，打印看看')
                print(walkthrough)
                break # 跳出循环
        else:
            action = model.next_action(state)
        counter += 1
        reward = game.act(action)
        final_reward += reward
        episode_sars.append((state, action, reward))
    txtLogger.add('===== 调整参数中 =====')
    model.train()
    baseline.train()
    G = 0  # 未来累计折扣奖励
    gamma = 0.95  # 折扣因子
    actor_losses = []
    critic_losses = []
    # 20250224 NOTE: 注意这里的逻辑必须是先clean_gradient，然后每一步backward_loss，最后来一个optimizer_step。如果将所有loss保留在数组里会导致内存不足。
    model.clean_gradient()
    actor_idle_steps = 0 # NOTE: 用来计算有多少step actor没有进行训练
    for state, action, instant_reward in reversed(episode_sars):
        G = instant_reward + gamma * G  # 计算回报
        # Advantage
        baseline_out = baseline.forward(state) # Need grad
        with torch.no_grad():
            advantage = G - baseline_out.item()
        # 更新actor NOTE: 2.26修改bug，如果acvantage小于零，我们不要更新actor，而只修正critic的预期
        if advantage <= 0:
            actor_losses.append(0)
            actor_idle_steps += 1
        else:
            actor_loss = model.action_select_loss(state, action, advantage) # NOTE: 这里是重点！！！
            model.backward_loss(actor_loss)
            actor_losses.append(actor_loss.item())
        # 更新critic
        target = torch.tensor([[G]], dtype=torch.float32).cuda()
        baseline_loss = squared_loss(target, baseline_out)
        baseline.update(baseline_loss)
        critic_losses.append(baseline_loss.item())
    model.optimizer_step()
    txtLogger.add('===== 一个episode训练完成 =====')
    txtLogger.write_txt_log()
    mean_actor_loss = sum(actor_losses) / len(episode_sars)
    mean_critic_loss = sum(critic_losses) / len(episode_sars)
    norm_score = game.get_score() / game.get_max_score()
    actor_idle_rate = actor_idle_steps / len(episode_sars)
    return norm_score, mean_actor_loss, mean_critic_loss, actor_idle_rate

def loss_scale(losses):
    return [min(num * 0.1, 2.0) for num in losses]

class Tester(bert_qac.Tester):
    def init_critic(self, bert = None, header = None):
        return SimpleBaseline(bert, header)
    def init_hook(self):
        self.txtLogger= Logger_simple('文字logger')
        self.init_params_for_log()
    def draw(self):
        a_loss = loss_scale(self.actor_losses)
        c_loss = loss_scale(self.critic_losses)
        # 需要对损失进行放大&缩小或者CLIP
        draw_line_chart(list(range(len(self.episode_rewards))), [a_loss, c_loss, self.valid_scores], ['a', 'c', 'r'])
    def maybe_valid_and_save(self):
        self.counter += 1
        need_valid_and_update = self.counter % 100 == 0
        if need_valid_and_update: # valid然后更新max_valid_score和last_valid_score，如果分数更大，保存checkpoint
            self.last_valid_score = batch_valid(self.actor.bert, save_readable = False)
            self.txtLogger.add(f'Valid分数为{self.last_valid_score}')
            self.txtLogger.write_txt_log()
            if self.last_valid_score > self.max_valid_score:
                self.max_valid_score = self.last_valid_score
                self.save_checkpoint()
        self.valid_scores.append(self.last_valid_score)
        if need_valid_and_update: # 绘制中间图像
            self.draw()
    def explore_only(self):
        from ftwp_info import all_train_game_paths
        from bert_for_ftwp import Game_for_rl
        paths = all_train_game_paths()
        game_count = len(paths)
        for index, path in enumerate(paths):
            print(f'{index}/{game_count}')
            game = Game_for_rl(path)
            r,a,c, actor_idle_rate =train_one_episode(self.actor, self.critic, game, None) # NOTE !!!
            self.episode_rewards.append(r)
            self.actor_losses.append(a)
            self.critic_losses.append(c)
            self.maybe_valid_and_save()
        self.txtLogger.write_txt_log()
    def traing_and_save(self):
        from ftwp_info import all_train_game_paths
        from bert_for_ftwp import Game_for_rl
        paths = all_train_game_paths()
        game_count = len(paths)
        for index, path in enumerate(paths):
            print(f'{index}/{game_count}')
            game = Game_for_rl(path)
            r,a,c, actor_idle_rate =train_one_episode(self.actor, self.critic, game, walkthrough=game.filter_walkthroughs_with_input_test()) # NOTE !!!
            self.txtLogger.add(f'Actor idle rate: {actor_idle_rate}')
            self.episode_rewards.append(r)
            self.actor_losses.append(a)
            self.critic_losses.append(c)
            self.maybe_valid_and_save()
        self.txtLogger.write_txt_log()
    def train_multiple_epoches(self, epoch = 3):
        self.counter = -1
        for i in range(epoch):
            self.traing_and_save()
    def explore_multiple_epoches(self, epoch = 4):
        self.counter = -1
        for i in range(epoch):
            self.explore_only()

