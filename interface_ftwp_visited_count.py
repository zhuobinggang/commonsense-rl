from interface_ftwp import Ftwp_interface_by_path, game_played_and_save_training_file, llm_auto_play
import common

class Ftwp_interface_visited_count(Ftwp_interface_by_path):
    def move_command_succeeded_callback(self, action_obs):
        # Added 2024.12.21 增加访问次数显示，在action history里面
        action, obs = super().move_command_succeeded_callback(action_obs)
        room_name = common.extract_room_name(obs)
        obs = obs + f' Visited {self.world_map[room_name]} times.'
        return action, obs
    

def game_for_test():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0()
    game = Ftwp_interface_visited_count(file_paths[1])
    game.verbose = True
    return game


# @history 2024.12.18 change dataset size from 10 to 20
def batch_training_file_prepare():
    from ftwp_info import train_set_v0
    file_paths = train_set_v0(10) 
    for file_path in file_paths:
        game = Ftwp_interface_visited_count(file_path)
        game_played_and_save_training_file(game, output_file_path = f'exp/auto_filename/{game.game_name}.jsonl')


def random_and_out_as_one():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare()


train_file_path = 'exp/ftwp_finetune_visited_count/training.jsonl'

def valid_then_upload():
    from finetune_simplify_desc import valid, finetune_file_upload
    valid(train_file_path)
    finetune_file_upload(train_file_path)

FILE_ID_10 = 'file-BmbKXJskpkJm4mbauzxkoG'

def finetune_ftwp():
    from finetune_simplify_desc import finetune
    finetune(FILE_ID_10)


def valid_then_upload_testing():
    from finetune_simplify_desc import valid, finetune_file_upload
    train_file_path = 'exp/auto_filename/test_training.jsonl'
    valid(train_file_path)
    finetune_file_upload(train_file_path)
    FILE_ID = 'file-RiuBCeZJj9fZSDvH6JKyTk'

def test_finetune():
    from finetune_simplify_desc import finetune
    finetune('file-RiuBCeZJj9fZSDvH6JKyTk')

# ================== 2025.1.6 测试 ====================

TUNED_MODELS = ['ft:gpt-4o-mini-2024-07-18:personal::AmYBh1o9:ckpt-step-143', 'ft:gpt-4o-mini-2024-07-18:personal::AmYBicFP:ckpt-step-286', 'ft:gpt-4o-mini-2024-07-18:personal::AmYBi3DL']
BEST = TUNED_MODELS[1]

def batch_valid():
    from ftwp_info import temp_test_valid_set
    _, valid_game_paths = temp_test_valid_set()
    for model in TUNED_MODELS:
        for game_index, game_path in enumerate(valid_game_paths):
            game = Ftwp_interface_visited_count(game_path)
            llm_auto_play(game, game_index, testing=False, gpt_type = model, max_try=20)

def run_test_temp():
    from ftwp_info import temp_test_valid_set
    test_game_paths, _ = temp_test_valid_set()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_interface_visited_count(game_path)
        llm_auto_play(game, game_index, testing=True, gpt_type = model, max_try=20)


def run_test():
    from ftwp_info import test_set_v0
    test_game_paths = test_set_v0()
    model = BEST
    for game_index, game_path in enumerate(test_game_paths):
        game = Ftwp_interface_visited_count(game_path)
        llm_auto_play(game, game_index, testing=True, gpt_type = model, max_try=20)


