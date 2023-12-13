from openai import OpenAI
from taku_step_by_step import get_game_env, ONE_SHOT_EXP
import taku_step_by_step as taku
import pyperclip
client = OpenAI()

def quest_gpt4(system_msg, user_msg):
    completion = client.chat.completions.create(
      # model="gpt-3.5-turbo",
      model="gpt-4-1106-preview",
      messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
      ]
    )
    print(completion.choices[0].message.content)
    print(completion.usage)
    # To clipboard
    text_to_paste = f'{completion.choices[0].message.content}\n{completion.usage}'
    pyperclip.copy(text_to_paste)
    return completion


def call_gpt4_and_print(enviroment, inventory, available_actions, action_obs_pairs = []):
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
    print(action_history)
    system_msg = f"""Task: {task}
Example walkthrough: {ONE_SHOT_EXP}
Action history: {action_history}
Inventory: {inventory}
Current enviroment: {enviroment}"""
    user_msg = f"""Action you can take: 
{available_action_text}
Question: To put things in there proper locations and improve your score, what should you do? Think step by step then choose 'one' action from above list.
Consideration: <fill in>
Next action: <fill in>"""
    return quest_gpt4(system_msg, user_msg)
    
def act_step_by_step(env, command = None, caller = call_gpt4_and_print):
    return taku.act_step_by_step(env, command, caller)
