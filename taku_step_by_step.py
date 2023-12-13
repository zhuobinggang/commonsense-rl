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






ONE_SHOT_EXP = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor.You can see a shoe cabinet. The shoe cabinet contains a dirty checkered shirt. You can make out a key holder. The key holder is ordinary. Unfortunately, there isn't a thing on it. What, you think everything in TextWorld should have stuff on it? You see a gleam over in a corner, where you can see an umbrella stand. But the thing is empty. You can see a hat rack. Looks like someone's already been here and taken everything off it, though. You smell an intriguing smell, and follow it to a coat hanger. But there isn't a thing on it.There is an exit to the west. Don't worry, there is no door.There is a dirty cardigan and a dirty maroon dress on the floor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room.An open washing machine is right there by you. The washing machine contains a dirty yellow T-shirt and a dirty yellow dress. You can see a clothes drier. The clothes drier contains a wet azure dress. You can make out an opened laundry basket. The laundry basket is empty, what a horrible day! Oh, great. Here's a suspended shelf. Looks like someone's already been here and taken everything off it, though. You make out a bench. But oh no! there's nothing on this piece of trash. You make out a work table. You shudder, but continue examining the work table. The work table is usual. The work table appears to be empty.There is an exit to the east. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point.                               *** The End ***You scored 7 out of a possible 7,"""

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



def act_step_by_step(env, command = None, printer = printer_step_by_step4):
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
    return printer(enviroment, inventory, available_actions, action_obs_pairs)
