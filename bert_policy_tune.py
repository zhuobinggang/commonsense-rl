# 2025.2.12 policy gradient算法
from bert_common import Abs_model_policy_gradient, Logger_loss_and_score
from common import Logger_simple, Fake_text_logger
import torch

class FakeChartLogger:
    def __init__(self, filename) -> None:
        pass
    def add_loss(self, loss):
        pass
    def add_reward(self, reward):
        pass
    def episode_log(self):
        pass

# TODO: 测试
# TODO: 爆改logger记录每一步的损失值？最好能够记录这次优化的是哪个action，并且记录对应的损失值，瞬时reward也记录一下
def train_one_episode(model: Abs_model_policy_gradient, game, walkthrough = None, txtLogger = None):
    if not txtLogger:
        txtLogger = Fake_text_logger('')
    # 先使用walkthrough训练一次，然后自主探索3次
    episode_sars = [] # state, action, reward
    game.reset()
    counter = 0
    final_reward = 0
    model.eval()
    while not any([counter >= 50, game.is_won(), game.is_lost()]):
        state = game.get_state()
        if walkthrough:
            if counter < len(walkthrough):
                action = walkthrough[counter]
            else:
                txtLogger.add('到头了，但是游戏没有结束，说明walkthrough可能有问题，打印看看')
                txtLogger.add(walkthrough)
                break # 跳出循环
        else:
            action = model.next_action(state)
        counter += 1
        reward = game.act(action)
        final_reward += reward
        episode_sars.append((state, action, reward))
    model.train()
    G = 0  # 未来累计折扣奖励
    gamma = 0.95  # 折扣因子
    losses = []
    # 20250224 NOTE: 注意这里的逻辑必须是先clean_gradient，然后每一步backward_loss，最后来一个optimizer_step。如果将所有loss保留在数组里会导致内存不足。
    model.clean_gradient()
    for state, action, instant_reward in reversed(episode_sars):
        G = instant_reward + gamma * G  # 计算回报
        loss = model.action_select_loss(state, action, G)
        model.backward_loss(loss)
        losses.append(loss.item())
    model.optimizer_step()
    total_loss = sum((losses)) # NOTE: 改版后未测试
    # 20250303记录步数超30的情况
    if counter >= 30:
        txtLogger.add('训练时步数超过30')
    txtLogger.write_txt_log()
    episode_mean_actor_loss = total_loss / len(episode_sars)
    norm_score = game.get_score() / game.get_max_score()
    return norm_score, episode_mean_actor_loss

