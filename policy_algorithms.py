# 2025.2.12 policy gradient算法
from bert_common import Abs_model_policy_gradient, squared_loss
from common import Logger_simple, Fake_text_logger
import torch

# Standard REINFORCE
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
            action = model.next_action_with_explore(state)
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


def train_one_episode_baseline(model: Abs_model_policy_gradient, baseline, game, walkthrough = None, txtLogger = None):
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

# 尝试减去至今为止的平均reward
def train_one_episode_simplest_baseline(model: Abs_model_policy_gradient, game, walkthrough = None, txtLogger = None, steps_limit = 50):
    if not txtLogger:
        txtLogger = Fake_text_logger('')
    # 先使用walkthrough训练一次，然后自主探索3次
    episode_sars = [] # state, action, reward
    game.reset()
    counter = 0
    final_reward = 0
    model.eval()
    while not any([counter >= steps_limit, game.is_won(), game.is_lost()]):
        state = game.get_state()
        if walkthrough:
            if counter < len(walkthrough):
                action = walkthrough[counter]
            else:
                txtLogger.add('到头了，但是游戏没有结束，说明walkthrough可能有问题，打印看看')
                txtLogger.add(walkthrough)
                break # 跳出循环
        else:
            action = model.next_action_with_explore(state)
        counter += 1
        reward = game.act(action)
        final_reward += reward
        episode_sars.append((state, action, reward))
    model.train()
    G = 0  # 未来累计折扣奖励
    gamma = 0.95  # 折扣因子
    losses = []
    # 20250224: 注意这里的逻辑必须是先clean_gradient，然后每一步backward_loss，最后来一个optimizer_step。如果将所有loss保留在数组里会导致内存不足。
    model.clean_gradient()
    for state, action, instant_reward in reversed(episode_sars):
        G = instant_reward + gamma * G  # 计算回报
        scalar = max(0.1, min(10, G / (model.get_average_G() + 1e-4))) # 逻辑: 如果当前的预期回报比平均预期回报更好就应该得到强调，否则会以分数的形式来减轻其影响。缩放的倍数在(0.1~10)之间。
        model.update_average_G(G)
        loss = model.action_select_loss(state, action, scalar) # 这里具体的计算是log_softmax，然后乘以scalar
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

# 尝试减去至今为止的平均reward
def train_one_episode_simplest_baseline_v2(model: Abs_model_policy_gradient, game, walkthrough = None, txtLogger = None, steps_limit = 50):
    if not txtLogger:
        txtLogger = Fake_text_logger('')
    # 先使用walkthrough训练一次，然后自主探索3次
    episode_sars = [] # state, action, reward
    game.reset()
    counter = 0
    final_reward = 0
    model.eval()
    while not any([counter >= steps_limit, game.is_won(), game.is_lost()]):
        state = game.get_state()
        if walkthrough:
            if counter < len(walkthrough):
                action = walkthrough[counter]
            else:
                txtLogger.add('到头了，但是游戏没有结束，说明walkthrough可能有问题，打印看看')
                txtLogger.add(walkthrough)
                break # 跳出循环
        else:
            action = model.next_action_with_explore(state)
        counter += 1
        reward = game.act(action)
        final_reward += reward
        episode_sars.append((state, action, reward))
    model.train()
    G = 0  # 未来累计折扣奖励
    gamma = 0.95  # 折扣因子
    losses = []
    # 20250224: 注意这里的逻辑必须是先clean_gradient，然后每一步backward_loss，最后来一个optimizer_step。如果将所有loss保留在数组里会导致内存不足。
    model.clean_gradient()
    for state, action, instant_reward in reversed(episode_sars):
        G = instant_reward + gamma * G  # 计算回报
        scalar = max(0.1, min(10, G / (model.get_average_G() + 1e-4))) # 逻辑: 如果当前的预期回报比平均预期回报更好就应该得到强调，否则会以分数的形式来减轻其影响。缩放的倍数在(0.1~10)之间。
        model.update_average_G(G)
        loss = model.action_select_loss(state, action, scalar) # 这里具体的计算是log_softmax，然后乘以scalar
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
