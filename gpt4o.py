from gpt4 import GPT_Caller, get_game_env, save_record

def run(game_idx = 0):
    env = get_game_env(2, game_idx)
    caller = GPT_Caller(env, zero_shot = False, gpt_type = 'gpt-4o-2024-05-13', cot = True, one_shot_easy = False, no_augment = False)
    caller.step(None) # first step
    return caller

#### 保存相邻房间的信息 ####
#### TODO: 直接填加一个字段叫做Another room:
#### TODO: 需要找到判断房间移动的准确根据, 大概从env里面可以找到？
#### TODO: 把step之后的info也保存到env里面，可能有重要信息

