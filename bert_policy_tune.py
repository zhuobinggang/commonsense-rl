# 2025.2.12 policy gradient算法

# TODO: 完善
def train_one_episode(model, game, walkthrough = None):
    # 先使用walkthrough训练一次，然后自主探索3次
    episode_sars = [] # state, action, reward
    game.reset()
    counter = 0
    final_reward = 0
    while not any([counter >= 30, game.is_won(), game.is_lost()]):
        state = game.get_state()
        action = walkthrough[counter] if walkthrough else model.next_action(state)
        counter += 1
        reward = game.act(action)
        final_reward += reward
        episode_sars.append((state, action, reward))
    decend_coefficient = 0.95
    decended_final_reward = final_reward
    for state, action, instant_reward in reversed(episode_sars):
        reward_scalar = instant_reward + decended_final_reward
        decended_final_reward = decend_coefficient * decended_final_reward # 最终奖励递减
        model.update_policy(state, action, reward_scalar)