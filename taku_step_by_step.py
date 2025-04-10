import taku
step = taku.step

# env = get_game_env(2, 0, 1)
def get_game_env(level_index = 0, game_index = 0, dataset_index = 1, printer = None, max_step = 50, need_reset = False):
    printer = None
    need_reset = False
    env = taku.get_game_env(level_index, game_index, dataset_index, printer, max_step, need_reset)
    env.meta_info = f'get_game_env({level_index},{game_index},{dataset_index})'
    env.meta_name = f'{taku.DATASET_NAMES[dataset_index]}_{taku.LEVELS[level_index]}_{game_index}'
    initiate_env(env)
    return env


def dd():
    env = get_game_env()
    action_obs_pairs = []
    command = ''
    obs, info = step(env, command, caller  = None)
    obs = obs[0].strip().replace('\n','')
    inventory = info['inventory'][0].strip().replace('\n','')
    # TODO: print(obs, infos)
    action_obs_pairs.append((command, obs))
    enviroment, _ = step(env, 'look', caller = None)
    enviroment = enviroment[0].strip().replace('\n','')
    available_actions = info['admissible_commands'][0]
    printer_step_by_step(enviroment, inventory, available_actions, action_obs_pairs)


def printer_step_by_step(enviroment, inventory, available_actions, action_obs_pairs = []):
    task = 'Your task is put things in there proper locations and improve your score.'
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} -> {obs} '
    else:
        action_history = 'No action was taken now.'
    template = f"""Task: {task}
Action history: {action_history}
Current enviroment: {enviroment}
Inventory: {inventory}
Next action you can take: {available_actions}

To complete your task, have you made any mistakes that hindered you from completing tasks more efficiently? Which action (only one) should you take next? Let's think step by step."""
    print(template)

def printer_step_by_step3(enviroment, inventory, available_actions, action_obs_pairs = []):
    task = 'You are a experienced text game player, your goal is put things in there proper locations and improve your score.'
    available_action_text = ''
    for act in available_actions:
        available_action_text += f'* {act}\n'
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} -> {obs} '
    else:
        action_history = 'No action was taken now.'
    template = f"""Task: {task}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}

Action you can take: 
{available_action_text}

Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>
"""
    print(template)

# A simple version
def printer_step_by_step2(enviroment, inventory, available_actions, action_obs_pairs = []):
    task = 'You are a experienced text game player, your goal is put things in there proper locations and improve your score.'
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} -> {obs} '
    else:
        action_history = 'No action was taken now.'
    template = f"""Task: {task}
Action history: {action_history}
Current enviroment: {enviroment}
Inventory: {inventory}

To complete your goal, think step by step then choose 'one' action from: {available_actions}"""
    print(template)






ONE_SHOT_EXP = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point."""
def printer_step_by_step4(enviroment, inventory, available_actions, action_obs_pairs = []):
    task = 'You are a experienced text game player, your goal is put things in there proper locations and improve your score.'
    available_action_text = ''
    for act in available_actions:
        available_action_text += f'* {act}\n'
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} -> {obs} '
    else:
        action_history = 'No action was taken now.'
    template = f"""Task: {task}
Example walkthrough: {ONE_SHOT_EXP}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}

Action you can take: 
{available_action_text}

Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>
"""
    print(template)

def initiate_env(env):
    if not hasattr(env, 'action_obs_pairs'):
        env.action_obs_pairs = []
    if not hasattr(env, 'system_user_msgs'): # 2024.5.22
        env.system_user_msgs = []
    if not hasattr(env, 'gpt_responses'): # 2024.5.22
        env.gpt_responses = []
    if not hasattr(env, 'available_actions'): # 2024.1.8 增加env.available_actions
        env.available_actions = []
    if not hasattr(env, 'last_reward'): # 2024.1.8 增加env.instant_reward
        env.last_reward = 0
        env.instant_reward = 0
    if not hasattr(env, 'score_by_step'): # 2024.2.9 增加env.score_by_step
        env.score_by_step = []
    if not hasattr(env, 'end'): # 2024.5.22
        env.end = False
    if not hasattr(env, 'info'): # 2024.5.23
        env.info = None # Will be injected in taku.step()

def act_step_by_step(env, command = None, caller = printer_step_by_step4):
    #initiate_env(env)
    action_obs_pairs = env.action_obs_pairs
    if command:
        obs, info = step(env, command, caller = None, need_reward = False)
        obs = obs[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        description = info['description']
        description = description[0].strip().replace('\n','')
        # 如果obs和description重复，应该被截断
        if obs == description:
            obs = obs.split('.')[0] + '.'
        action_obs_pairs.append((command, obs))
        available_actions = info['admissible_commands'][0]
        env.counter_taku += 1
        # TODO: 记录瞬时奖励
        new_reward = info['score'][0]
        env.instant_reward = new_reward - env.last_reward
        env.last_reward = new_reward
        if env.instant_reward != 0:
            print(f'RECORDED REWARD: {env.instant_reward}')
    else: # 重新开始的情况
        # print('RESTAR\n\n')
        _, info = env.reset()
        description = info['description']
        description = description[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        available_actions = info['admissible_commands'][0]
        action_obs_pairs = []
        env.counter_taku = 0
        env.last_reward = 0
        env.instant_reward = 0
    env.available_actions = available_actions
    if info['won'][0]:
        # 打引结束信息
        # print(f"YOU WIN, score at {info['score']}/{info['max_score']}, steps {info['moves']}")
        action_history = ''
        if len(action_obs_pairs) > 0:
            for idx, (act, obs) in enumerate(action_obs_pairs):
                action_history += f'Action {idx}: {act} -> {obs} '
        print(action_history)
    else:
        return caller(description, inventory, available_actions, action_obs_pairs)

# 2024.01.21 观察增强，如果是放错位置要有足够的提示
def is_placing_item(command):
    command = command.lower()
    return command.split()[0] in ['put', 'insert']
RIGHT_POSITION = 'Right position.'
WRONG_POSITION = 'Wrong position, you should put it somewhere else, maybe the other room.'
def act_step_by_step_obs_augment(env, command = None, no_augment = False):
    #initiate_env(env)
    action_obs_pairs = env.action_obs_pairs
    if command:
        obs, info = step(env, command, caller = None, need_reward = False)
        obs = obs[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        enviroment = info['description']
        enviroment = enviroment[0].strip().replace('\n','')
        # 如果obs和enviroment重复，应该被截断
        if obs == enviroment:
            obs = obs.split('.')[0] + '.'
        available_actions = info['admissible_commands'][0]
        # TODO: 记录瞬时奖励
        new_reward = info['score'][0]
        env.instant_reward = new_reward - env.last_reward
        env.last_reward = new_reward
        # 2024.1.21 如果放错位置应该提供足够的提示
        if no_augment:
            print('NO AUGMENT')
            pass
        else:
            if env.instant_reward != 0:
                print(f'RECORDED REWARD: {env.instant_reward}')
                obs += RIGHT_POSITION
            else:
                if is_placing_item(command):
                    obs_lower = obs.lower()
                    if obs_lower.startswith("you can't") or obs_lower.startswith('you need'):
                        pass
                    else:
                        obs += WRONG_POSITION
        action_obs_pairs.append((command, obs))
    else: # 重新开始的情况
        # print('RESTAR\n\n')
        _, info = env.reset()
        enviroment = info['description']
        enviroment = enviroment[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        available_actions = info['admissible_commands'][0]
        action_obs_pairs = []
        env.last_reward = 0
        env.instant_reward = 0
    env.available_actions = available_actions
    if info['won'][0]:
        env.end = True
        # 打引结束信息
        # print(f"YOU WIN, score at {info['score']}/{info['max_score']}, steps {info['moves']}")
    return enviroment, inventory, available_actions, action_obs_pairs

def act_step_by_step_obs_augment_manually(env, command = None):
    enviroment, inventory, available_actions, action_obs_pairs = act_step_by_step_obs_augment(env, command)
    print(enviroment)
    print(inventory)
    for item in available_actions:
        print(item)


# 2024.01.18 返回所有必要信息而不是调用回调函数
def act_step_by_step_simple(env, command = None):
    if not hasattr(env, 'action_obs_pairs'):
        env.action_obs_pairs = []
    if not hasattr(env, 'available_actions'): # 2024.1.8 增加env.available_actions
        env.available_actions = []
    if not hasattr(env, 'last_reward'): # 2024.1.8 增加env.instant_reward
        env.last_reward = 0
        env.instant_reward = 0
    action_obs_pairs = env.action_obs_pairs
    if command:
        obs, info = step(env, command, printer = None, need_reward = False)
        obs = obs[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        enviroment = info['description']
        enviroment = enviroment[0].strip().replace('\n','')
        # 如果obs和enviroment重复，应该被截断
        if obs == enviroment:
            obs = obs.split('.')[0] + '.'
        action_obs_pairs.append((command, obs))
        available_actions = info['admissible_commands'][0]
        # TODO: 记录瞬时奖励
        new_reward = info['score'][0]
        env.instant_reward = new_reward - env.last_reward
        env.last_reward = new_reward
        if env.instant_reward != 0:
            print(f'RECORDED REWARD: {env.instant_reward}')
    else: # 重新开始的情况
        # print('RESTAR\n\n')
        _, info = env.reset()
        enviroment = info['description']
        enviroment = enviroment[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        available_actions = info['admissible_commands'][0]
        action_obs_pairs = []
        env.last_reward = 0
        env.instant_reward = 0
    env.available_actions = available_actions
    if info['won'][0]:
        # 打引结束信息
        # print(f"YOU WIN, score at {info['score']}/{info['max_score']}, steps {info['moves']}")
        pass
    return enviroment, inventory, available_actions, action_obs_pairs


# 2024.1.8 : 将上一步的available action list保存在env里边，command用序号输入
def printer_command_with_index(enviroment, inventory, available_actions, action_obs_pairs = []):
    available_action_text = ''
    for idx, act in enumerate(available_actions):
        available_action_text += f'[{idx}] {act}\n'
    action_history = ''
    if len(action_obs_pairs) > 0:
        for idx, (act, obs) in enumerate(action_obs_pairs):
            action_history += f'Action {idx}: {act} -> {obs} '
    else:
        action_history = ''
    template = f"""Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}
Action you can take: 
{available_action_text}
"""
    # print(template)
    return template


# 2024.1.8 : 将上一步的available action list保存在env里边，command用序号输入
def interact_with_env(env, command_idx = -1, printer = printer_command_with_index):
    command = None if command_idx == -1 else env.available_actions[command_idx]
    return act_step_by_step(env, command, printer)

