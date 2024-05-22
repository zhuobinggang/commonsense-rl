from openai import OpenAI
from taku_step_by_step import get_game_env, ONE_SHOT_EXP, act_step_by_step_obs_augment
import pyperclip
client = OpenAI()

ONE_SHOT_EXP_AUGMENT = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point.Right position. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point.Right position. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point.Right position. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point.Right position. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point.Right position. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point.Right position. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point.Right position."""


ONE_SHOT_EXP_NO_AUG = """Action 0: insert dirty yellow dress into washing machine -> You put the dirty yellow dress into the washing machine.Your score has just gone up by one point. Action 1: take dirty yellow T-shirt from bench -> You take the dirty yellow T-shirt from the bench. Action 2: insert dirty yellow T-shirt into washing machine -> You put the dirty yellow T-shirt into the washing machine.Your score has just gone up by one point. Action 3: take wet azure dress from suspended shelf -> You take the wet azure dress from the suspended shelf. Action 4: insert wet azure dress into clothes drier -> You put the wet azure dress into the clothes drier.Your score has just gone up by one point. Action 5: take white cap from bench -> You take the white cap from the bench. Action 6: go east -> -= Corridor =-You've entered a corridor. Action 7: put white cap on hat rack -> You put the white cap on the hat rack.Your score has just gone up by one point. Action 8: take dirty cardigan -> You pick up the dirty cardigan from the ground. Action 9: take dirty checkered shirt from shoe cabinet -> You take the dirty checkered shirt from the shoe cabinet. Action 10: take dirty maroon dress -> You pick up the dirty maroon dress from the ground. Action 11: go west -> -= Laundry Room =-You've entered a laundry room. Action 12: insert dirty cardigan into washing machine -> You put the dirty cardigan into the washing machine.Your score has just gone up by one point. Action 13: insert dirty checkered shirt into washing machine -> You put the dirty checkered shirt into the washing machine.Your score has just gone up by one point. Action 14: insert dirty maroon dress into washing machine -> You put the dirty maroon dress into the washing machine.Your score has just gone up by one point."""

ONE_SHOT_EXP_AUGMENT_SIMPLE = """Action 0: take dirty gray underpants from work table -> You take the dirty gray underpants from the work table. Action 1: insert dirty gray underpants into washing machine -> You put the dirty gray underpants into the washing machine.Your score has just gone up by one point.Right position."""


ONE_SHOT_EXP_SIMPLE_NO_AUG = """Action 0: take dirty gray underpants from work table -> You take the dirty gray underpants from the work table. Action 1: insert dirty gray underpants into washing machine -> You put the dirty gray underpants into the washing machine.Your score has just gone up by one point."""

def quest_gpt4(system_msg, user_msg, gpt3 = False):
    if gpt3:
        model="gpt-3.5-turbo"
    else:
        model = "gpt-4-1106-preview"
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
      ]
    )
    content = completion.choices[0].message.content
    usage = str(completion.usage)
    print(content)
    print(usage)
    # To clipboard
    # NOTE: 自动提取command
    for line in content.split('\n'):
        line = line.lower()
        if line.startswith('next action:'):
            text_to_paste = line.replace('next action:', '').strip()
            pyperclip.copy(text_to_paste)
            print(f'COMMAND GOT: {text_to_paste}')
    dic = {'response': content, 'usage': usage}
    return dic

def call_gpt4_and_print_raw(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, gpt3 = False, question_template = None, ONE_SHOT_EASY = False, no_augment = False):
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
    if zero_shot:
        system_msg = f"""Task: {task}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}"""
    else:
        if no_augment:
            walkthrough = ONE_SHOT_EXP_SIMPLE_NO_AUG if ONE_SHOT_EASY else ONE_SHOT_EXP_NO_AUG
        else: 
            walkthrough = ONE_SHOT_EXP_AUGMENT_SIMPLE if ONE_SHOT_EASY else ONE_SHOT_EXP_AUGMENT
        system_msg = f"""Task: {task}
Example walkthrough: {walkthrough}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}"""
    user_msg = f"""Action you can take: 
{available_action_text}
{question_template}"""
    x = system_msg + '\n' + user_msg
    print(x)
    y = quest_gpt4(system_msg, user_msg, gpt3 = gpt3)
    return x, y

#### 2024.1.25 ablation expriment
def call_gpt4_and_print_sub_COT(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, gpt3 = False, ONE_SHOT_EASY = False, no_augment = False):
    # NOTE: -COT 删除了部分提示
    question_template = """Question: To put things in there proper locations and improve your score, what should you do? Choose 'one' action from above list.
Next action: <fill in>"""
    return call_gpt4_and_print_raw(enviroment, inventory, available_actions, action_obs_pairs, zero_shot, gpt3, question_template, ONE_SHOT_EASY, no_augment)


def call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs = [], zero_shot = True, gpt3 = False, ONE_SHOT_EASY = False, no_augment = False):
    question_template = f"""Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>"""
    return call_gpt4_and_print_raw(enviroment, inventory, available_actions, action_obs_pairs, zero_shot, gpt3, question_template, ONE_SHOT_EASY, no_augment)
    

def interact_with_env_raw(env, command = None, gpt3 = False, cot = True, zero_shot = True, ONE_SHOT_EASY = False, no_augment = False):
    if not hasattr(env, 'dataset'): # 2024.1.8 增加env.dataset
        env.dataset = []
    if not hasattr(env, 'score_by_step'): # 2024.2.9 增加env.score_by_step
        env.score_by_step = []
    enviroment, inventory, available_actions, action_obs_pairs = act_step_by_step_obs_augment(env, command, no_augment)
    env.score_by_step.append(env.last_reward)
    if not hasattr(env, 'end'):
        if cot:
            x, y = call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs, zero_shot = zero_shot, gpt3 = gpt3, ONE_SHOT_EASY = ONE_SHOT_EASY, no_augment = no_augment)
        else:
            x, y = call_gpt4_and_print_sub_COT(enviroment, inventory, available_actions, action_obs_pairs, zero_shot = zero_shot, gpt3 = gpt3, ONE_SHOT_EASY = ONE_SHOT_EASY, no_augment = no_augment)
        env.dataset.append((x,y))

def interact_with_env(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = False, cot = True, zero_shot = True)

def interact_with_env_gpt3(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = True, cot = True, zero_shot = True)

def interact_with_env_without_cot(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = False, cot = False, zero_shot = True)

def interact_with_env_oneshot(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = False, cot = True, zero_shot = False)

def interact_with_env_oneshot_simple(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = False, cot = True, zero_shot = False, ONE_SHOT_EASY = True)


## 2.9 START
def interact_with_env_oneshot_without_cot(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = False, cot = False, zero_shot = False)

def interact_with_env_oneshot_without_fa(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = False, cot = True, zero_shot = False, no_augment = True)

def interact_with_env_oneshot_without_gpt4(env, command = None):
    return interact_with_env_raw(env, command, gpt3 = True, cot = True, zero_shot = False)
## 2.9 END 

def save_dataset_raw(env, filename):
    dataset = env.dataset
    # 处理错位问题
    dic = {'env_meta': env.meta_info, 'dataset': dataset, 'scores': env.score_by_step}
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_dataset(env):
    save_dataset_raw(env, 'exp/gpt4_zero_shot_obs_augment/' + env.meta_name + '.pickle')

def save_dataset_oneshot(env, filename = None):
    save_dataset_raw(env, 'exp/gpt4_one_shot_obs_augment/' + env.meta_name + '.pickle')

def save_dataset_gpt35(env, filename = None):
    save_dataset_raw(env, 'exp/gpt35_augment/' + env.meta_name + '.pickle')

def save_dataset_without_cot(env, filename = None):
    save_dataset_raw(env, 'exp/gpt4_augment_without_cot/' + env.meta_name + '.pickle')

def load_dataset(filename):
    import pickle
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def save_dataset_record_score(env):
    save_dataset_raw_with_score_by_step(env, 'exp/gpt4_zero_shot_obs_augment_with_score_by_step/' + env.meta_name + '.pickle')
