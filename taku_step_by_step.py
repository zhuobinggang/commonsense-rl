import taku
step = taku.step

def get_game_env(level_index = 0, game_index = 0, dataset_index = 1, printer = None, max_step = 50, need_reset = False):
    printer = None
    need_reset = False
    env = taku.get_game_env(level_index, game_index, dataset_index, printer, max_step, need_reset)
    env.counter_taku = 0
    return env


def dd():
    env = get_game_env()
    action_obs_pairs = []
    command = ''
    obs, info = step(env, command, printer = None)
    obs = obs[0].strip().replace('\n','')
    inventory = info['inventory'][0].strip().replace('\n','')
    # TODO: print(obs, infos)
    action_obs_pairs.append((command, obs))
    enviroment, _ = step(env, 'look', printer = None)
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


# A simple version
def printer_step_by_step2(enviroment, inventory, available_actions, action_obs_pairs = []):
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
To complete your task, which action (only one) should you take next? Let's think step by step."""
    print(template)


def act_step_by_step(env, command = None, printer = printer_step_by_step2):
    if not hasattr(env, 'action_obs_pairs'):
        env.action_obs_pairs = []
    action_obs_pairs = env.action_obs_pairs
    if command:
        obs, info = step(env, command, printer = None)
        obs = obs[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        # TODO: print(obs, infos)
        action_obs_pairs.append((command, obs))
        enviroment, _ = step(env, 'look', printer = None)
        enviroment = enviroment[0].strip().replace('\n','')
        available_actions = info['admissible_commands'][0]
        env.counter_taku += 1
    else: # 重新开始的情况
        print('RESTAR\n\n')
        _ = env.reset()
        enviroment, info = step(env, 'look', printer = None)
        enviroment = enviroment[0].strip().replace('\n','')
        inventory = info['inventory'][0].strip().replace('\n','')
        available_actions = info['admissible_commands'][0]
        action_obs_pairs = []
        env.counter_taku = 0
    printer(enviroment, inventory, available_actions, action_obs_pairs)
