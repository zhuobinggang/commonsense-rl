# Baseline implementation for BERT-based REINFORCE(policy gradient)

import torch
from bert_qac import Critic, encode_get_cls
import bert_qac
from bert_common import Abs_model_policy_gradient, batch_valid
from common import draw_line_chart, Logger_simple
from policy_algorithms import Fake_text_logger
from policy_algorithms import train_one_episode_baseline

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
            r,a,c, actor_idle_rate =train_one_episode_baseline(self.actor, self.critic, game, None) # NOTE !!!
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
            r,a,c, actor_idle_rate =train_one_episode_baseline(self.actor, self.critic, game, walkthrough=game.filter_walkthroughs_with_input_test()) # NOTE !!!
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

