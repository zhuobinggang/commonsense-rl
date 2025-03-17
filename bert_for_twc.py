from typing import Any
from env import Env_extra_info
import common
import abstract_game_interface
from functools import lru_cache
import bert_behavior_clone

@lru_cache(maxsize=None)
def get_all_game_paths(root_dir = 'games/twc', dataset_type = 'train'):
    import os
    """
    获取 root_dir 目录下所有 train 文件夹中的 .ulx 文件的绝对路径。

    :param root_dir: 根目录，包含 easy, medium, hard 目录
    :return: 以 .ulx 结尾的所有训练集文件的绝对路径列表
    """
    ulx_files = []
    
    # 遍历 easy, medium, hard 目录
    for difficulty in ["easy", "medium", "hard"]:
        train_dir = os.path.join(root_dir, difficulty, dataset_type)
        
        # 确保 train 目录存在
        if os.path.exists(train_dir) and os.path.isdir(train_dir):
            # 遍历 train 目录下的所有文件
            for file in os.listdir(train_dir):
                if file.endswith(".ulx"):
                    ulx_files.append(os.path.abspath(os.path.join(train_dir, file)))
    
    return ulx_files

def get_all_train_paths():
    return get_all_game_paths(dataset_type = 'train')

def get_all_valid_paths():
    return get_all_game_paths(dataset_type = 'valid')

def get_all_test_paths():
    return get_all_game_paths(dataset_type = 'test')

def load_train_game(idx):
    paths = get_all_train_paths()
    return Game_interface(paths[idx])

class Env_twc_by_path(Env_extra_info):
    def __init__(self, game_path, no_augment = True) -> None:
        env = self.get_game_env_by_path(game_path)
        env.meta_info = f'{game_path}'
        env.meta_name = f'{game_path}'
        self.initiate_env(env)
        self.env = env
        self.no_augment = True;
        self.info = None


# NOTE: 2025.2.10 不使用room description，实验性质
def final_prompt_twc(room_name = '', action_history = '', action_list = ''):
    user_msg = ''
    user_msg += f"Room: {room_name if room_name else 'Unknown now'}\n"
    # user_msg += f"Recipe: {self.recipe if self.recipe else 'Unknown now'}\n"
    user_msg += f"Action history: {action_history if action_history else ''}\n" 
    user_msg += f'Available actions:\n{action_list}\n' if action_list else ''
    user_msg += 'Next action: [MASK]'
    return user_msg

class Game_interface(abstract_game_interface.Game_interface):
    def __init__(self, game_path, no_augment = True): # datset_index = 0 for training, hard_level_index = 2 for hard-level.
        self.game_path = game_path
        game_name = game_path.split('/')[-1]
        self.game_name = game_name
         # train set
        self.env = Env_twc_by_path(game_path, no_augment)
        self.init_all_params()
        self.init_hook()
        self.verbose = True # NOTE: 注意更改
    def init_all_params(self):
        self.dataset_index = 'trainset'
        self.hard_level_index = 'unknown'
        self.finetune_triples = [] # (sys, usr, command_next)
        self.current_sys = ''
        self.current_usr = ''
        self.command = ''
        self.updated_description = ''
        self.another_room_info = 'Unknown'
        self.filename = f'TWC_{self.game_name}.json'
        self.won = False
        self.lost = False
        self.world_map = {} # 2024.12.21 用于存储访问过的地点次数
        self.desc_update_cache = {} # 2025.1.7 储存desc更新
        self.recipe = '' # 2025.1.13 储存菜谱
    def reset(self):
        self.act_and_output(None)
        self.init_all_params()
        self.init_hook()
    def init_hook(self):
        self.filter_startword_list = ['examine', 'close', 'eat', 'look', 'drop', 'inventory']
        self.walkthrough_with_meta_datas = [] # 2025.2.11 用于记录walkthrough
        self.walkthrough_path = self.game_path.replace('.ulx', '.walkthrough')
    def save_walkthrough(self):
        common.save_pickle_file(self.walkthrough_path, self.walkthrough_with_meta_datas)
    def load_walkthrough(self):
        return common.load_dataset(self.walkthrough_path)
    def get_walkthrough(self):
        if len(self.walkthrough_with_meta_datas) < 1:
            self.walkthrough_with_meta_datas = self.load_walkthrough()
        return [act for act, _ in self.walkthrough_with_meta_datas]
    def construct_sys_usr(self):
        description, inventory, available_actions, action_obs_pairs = self.description, self.inventory, self.available_actions, self.action_obs_pairs
        room_name = common.extract_room_name(description)
        action_history = common.action_obs_pairs_to_history(action_obs_pairs, seperator='>')
        action_list = common.actions_to_list_number(available_actions)
        usr = final_prompt_twc(room_name, action_history, action_list)
        self.x = usr
        sys = ''
        return sys, usr
    def print_walkthrough(self): # TODO: 打印时候要高亮得分项目
        for act, meta in self.walkthrough_with_meta_datas:
            txt = f'{act}' + (' [O]' if meta['score_up'] else '')
            print(txt)
    def action_obs_pairs_got_callback(self, action_obs_pairs):
        if not action_obs_pairs or len(action_obs_pairs) < 1:
            return
        act, obs = action_obs_pairs[-1]
        meta = {}
        if 'Your score has just gone up by one point' in obs:
            meta['score_up'] = True
        else:
            meta['score_up'] = False
        self.walkthrough_with_meta_datas.append((act, meta))
    def input(self, command):
        self.command = command
        command_idx = self.filtered_commands.index(command)
        self.finetune_triples.append((self.x, command_idx))
        self.act_and_output(command)
    def save_as_json(self):
        import json
        f = open(f'exp/auto_filename/{self.game_name}.jsonl', 'w')
        for x, y in self.finetune_triples:
            obj = {'x': x.strip(), 'y': str(y).strip()}
            line = json.dumps(obj)
            f.write(line + '\n')
        f.close()
    def back(self):
        actions = self.get_walkthrough()
        actions = actions[:-1]
        self.reset()
        self.auto_play(actions)
    def get_x(self):
        return self.x
    def save_readable(self):
        f = open(f'exp/auto_filename/{self.filename}.txt', 'w')
        for x, y in self.finetune_triples:
            f.write(f'{x}\n{y}\n\n')
        f.close()


# ================ 准备训练文件 ===================

def training_files_prepare():
    paths = get_all_train_paths()
    for path in paths:
        game = Game_interface(path)
        walkthrough = game.get_walkthrough()
        game.auto_play(walkthrough)
        game.save_as_json()

def final_training_file_prepare():
    from finetune_simplify_desc import lines_random_train_file_prepare
    lines_random_train_file_prepare(directory_path = 'exp/auto_filename', out_path='exp/auto_filename/twc_for_bert.jsonl')

# =============== 正式训练 =====================

def batch_valid(model, tokenizer = None, save_readable = True, test_game_paths = [], file_prefix = ''):
    import bert_for_ftwp
    if not tokenizer:
        tokenizer = bert_for_ftwp.default_tokenizer()
    if len(test_game_paths) < 1:
        test_game_paths = get_all_valid_paths()
    scores = []
    max_scores = []
    for path in test_game_paths:
        game = Game_interface(path)
        game.verbose = False
        game.filename = file_prefix + game.filename # Added 2025.2.8
        score, max_score = bert_for_ftwp.trained_model_autoplay(game, model, tokenizer, save_readable)
        scores.append(score)
        max_scores.append(max_score)
    return sum(scores) / sum(max_scores)

class LoggerTWC(bert_behavior_clone.Logger):
    def cal_valid_score_and_save_checkpoint(self):
        self.model.eval() # 进入校验模式
        norm_score = batch_valid(self.model, save_readable=False)
        self.model.train() # 返回训练模式
        if norm_score > self.max_score + self.min_delta:
            self.text_log += f'更新max score = {norm_score}\n'
            self.save_checkpoint()
            self.max_score = norm_score
            self.update_checkpoint_timestep = self.global_counter
        elif norm_score > self.max_score:
            self.text_log += f'没有超过阈值{self.min_delta}！{norm_score} ~= {self.max_score}\n'
        scalar = 10
        return norm_score * scalar

def twc_dataloader(batch = 4):
    from torch.utils.data import DataLoader
    from bert_behavior_clone import CustomDatasetTWC
    dl = DataLoader(CustomDatasetTWC(), batch_size = batch)
    return dl

def train(log_filename = '', epoch = 1):
    from transformers import AutoTokenizer, ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBertForMaskedLM.from_pretrained(model_id)
    model = model.train()
    from bert_behavior_clone import train_loop
    BATCH = 4
    dataLoader = twc_dataloader(batch = BATCH)
    logger = LoggerTWC(model, batch_size=BATCH, log_step=50)
    train_loop(model, tokenizer, log_filename=log_filename, batch = BATCH, epoch = epoch, dataloader=dataLoader, logger=logger)
    return model, tokenizer

# 实验结果已经出来了, 查看docs/ftwp_bert_baseline，epoch设定为2即可
def early_stop_exp(epoch = 10):
    return train(log_filename='early_stop_exp', epoch = epoch)


# =================== 测试 ===========================

def batch_test():
    import bert_for_ftwp
    model, toker = bert_for_ftwp.load_trained_model('/home/taku/Downloads/TWC_exp/exp_0211/default.tch')
    test_game_paths = get_all_test_paths()
    return batch_valid(model, toker, test_game_paths=test_game_paths)

def test_test():
    import bert_for_ftwp
    model, toker = bert_for_ftwp.load_trained_model('/home/taku/Downloads/TWC_exp/exp_0211/default.tch')
    valid_paths = get_all_valid_paths()
    game = Game_interface(valid_paths[0])
    game.reset()
    model.eval()
    return bert_for_ftwp.get_command_probs(game, toker, model)