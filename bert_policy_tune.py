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
def train_one_episode(model: Abs_model_policy_gradient, game, walkthrough = None, txtLogger = None, chartLogger = None):
    if not txtLogger:
        # txtLogger = Logger_simple('测试用文字logger')
        txtLogger = Fake_text_logger('')
    if not chartLogger:
        # chartLogger = Logger_loss_and_score('测试用图像logger')
        chartLogger = FakeChartLogger('')
    # 先使用walkthrough训练一次，然后自主探索3次
    episode_sars = [] # state, action, reward
    game.reset()
    counter = 0
    final_reward = 0
    model.eval()
    txtLogger.add('===== 探索中 =====')
    while not any([counter >= 30, game.is_won(), game.is_lost()]):
        state = game.get_state()
        try:
            action = walkthrough[counter] if walkthrough else model.next_action(state)
        except:
            txtLogger.add('重大问题，counter大于walkthrougn长度，这是怎么回事？')
            txtLogger.add(walkthrough)
            txtLogger.add(counter)
            txtLogger.add('在遇到不存在的action时候应该跳过才对，不需要执行')
        counter += 1
        reward = game.act(action)
        final_reward += reward
        if reward:
            # txtLogger.add(f'{state.x}\n')
            # txtLogger.add(f'{"[+1]" if reward else ""} {action}')
            pass
        episode_sars.append((state, action, reward))
    txtLogger.add('===== 调整参数中 =====')
    model.train()
    decend_coefficient = 0.95
    decended_final_reward = final_reward
    model.clean_gradient()
    losses = []
    for state, action, instant_reward in reversed(episode_sars):
        reward_scalar = instant_reward + decended_final_reward
        decended_final_reward = decend_coefficient * decended_final_reward # 最终奖励递减
        # txtLogger.add(f'{action} 对应的reward_scalar为 {reward_scalar}')
        loss = model.action_select_loss(state, action, reward_scalar)
        losses.append(loss)
        loss_number = loss.item()
        chartLogger.add_loss(loss_number)
        chartLogger.add_reward(instant_reward)
    total_loss = torch.sum(torch.stack(losses)) # NOTE: 改版后未测试
    txtLogger.add('===== 训练完成 =====')
    txtLogger.write_txt_log()
    chartLogger.episode_log()
    model.update_policy(total_loss)

