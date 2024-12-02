from cot_distill import game_played_then_request_reason_abstract, quest_get_reason, dics_random_train_file_prepare
import common
import global_variable as G
   

class Prompt_builder_for_cot_distll_v2:
    def __init__(self):
        self.task = G.TASK_COT_DISTLL
        self.action_history = ''
        self.inventory = ''
        self.another_room_info = ''
        self.current_enviroment = ''
        self.action_list = ''
        # NOTE
        self.best_action = ''
        # others
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''

    def build(self):
        system_msg = ''
        system_msg += f'Task: {self.task}\n' if self.task else ''
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += f'Action history: {self.action_history}\n' if self.action_history else ''
        user_msg += f'Inventory: {self.inventory}\n' if self.inventory else ''
        user_msg += f'Another room: {self.another_room_info}\n' if self.another_room_info else ''
        user_msg += f'Current environment: {self.current_enviroment}\n' if self.current_enviroment else ''
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += f'Reason (please provide the reasoning for why [{self.best_action}] is the best choice based on the observed state above, ending with "Therefore, I believe [{self.best_action}] is the best choice" while keeping it concise):'
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'


def sys_usr_cot_distill(description, inventory, available_actions, action_obs_pairs, another_room_info, best_action):
    promptor = Prompt_builder_for_cot_distll_v2()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs)
    promptor.inventory = inventory
    promptor.current_enviroment = description
    promptor.action_list = common.actions_to_list(available_actions)
    promptor.another_room_info = another_room_info# TODO
    promptor.best_action = best_action
    promptor.build()
    return promptor.system_msg, promptor.user_msg

def dic_to_sys_usr(dic):
    if not 'command' in dic:
        return None, None
    return sys_usr_cot_distill(dic['description'], dic['inventory'], dic['available_actions'], dic['action_obs_pairs'], dic['another_room_info'], dic['command'])


def save_pkl_with_game_index(game_index, pkl):
    import pickle
    with open(f'exp/auto_filename/game_{game_index}_human_play_4omini_reason_concisely.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(pkl, outp, pickle.HIGHEST_PROTOCOL)

def batch(start = 1, end = 5):
    for i in range(start, end):
        game_played_then_request_reason_abstract(i, dic_to_sys_usr, quest_get_reason, save_pkl_with_game_index)

def test(): 
    game_played_then_request_reason_abstract(1, dic_to_sys_usr, quest_get_reason, save_pkl_with_game_index)


def train_file_prepare():
    dics_random_train_file_prepare('exp/auto_filename', 'exp/auto_filename/cot_distill_training_v2.jsonl')

# ### validation
training_data_path ='exp/finetune_4omini_cot_distill_new/cot_distill_training_v2.jsonl'

def valid():
    from exp.finetune_4omini.varidation import load_ds, format_valid, ds_info, cost_est
    data_path = training_data_path
    ds = load_ds(data_path)
    format_valid(ds)
    dd = ds_info(ds)
    cost_est(ds, dd)


# Result: file-4ABmkrsyPrVYxQ5r2vyBLq
def finetune_file_upload():
    from exp.finetune_4omini.finetune import create_file
    return create_file(training_data_path)

UPLOADED_FILE_ID = 'file-4ABmkrsyPrVYxQ5r2vyBLq'

def finetune():
    from exp.finetune_4omini.finetune import fine_tune
    return fine_tune(UPLOADED_FILE_ID)

E1 = 'ft:gpt-4o-mini-2024-07-18:personal::AZqq53xr:ckpt-step-70'
E2 = 'ft:gpt-4o-mini-2024-07-18:personal::AZqq638q:ckpt-step-140'
E3 = 'ft:gpt-4o-mini-2024-07-18:personal::AZqq6wkl'
MODELS = [E1, E2, E3]

# ================= GAME PLAY ===================

from finetuned_play_cot_distill import llm_auto_play_valid_set, llm_auto_play

def batch_valid(start = 0, end = 1):
    for batch_index in range(start, end):
        for e in range(3):
            for game_index in range(5):
                _ = llm_auto_play_valid_set(game_index, f'B{batch_index}_', gpt_type=MODELS[e])


def batch_test(start = 0, end = 1):
    for batch_index in range(start, end):
        for game_index in range(5):
            _ = llm_auto_play(game_index, f'B{batch_index}_', gpt_type=E3)