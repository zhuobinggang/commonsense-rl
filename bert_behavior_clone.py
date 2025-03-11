import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import common
from bert_common import get_loss, get_optimizer, Game_for_rl
import bert_common
from common import draw_line_chart
from functools import lru_cache

class CustomDataset(Dataset):
    def __init__(self):
        self.init_dataframe()
    def init_dataframe(self):
        import pandas as pd
        self.dataframe = pd.read_json(path_or_buf='/home/taku/Downloads/cog2019_ftwp/procceeded_training_files/bert_trains.jsonl' ,lines=True)
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        return self.dataframe.x.iloc[idx], self.dataframe.y.iloc[idx]
    
class CustomDatasetTWC(CustomDataset):
    def init_dataframe(self):
        import pandas as pd
        self.dataframe = pd.read_json(path_or_buf='exp/train_files/twc_for_bert.jsonl' ,lines=True)

def get_dataloader(batch = 4):
    from torch.utils.data import DataLoader
    ds = CustomDataset()
    dl = DataLoader(ds, batch_size = batch)
    return dl


class MyDataLoader:
    def __init__(self):
        self.counter = 0
        self.ds = CustomDataset()
    def next(self):
        item = self.ds[self.counter]
        self.counter += 1
        if self.counter >= len(self.ds):
            self.counter = 0
        return item
    


class Model_behavior_clone(nn.Module):
    def __init__(self, loss_step_interval = 4, valid_step_interval = 5000, prefix = 'bert_behavior_clone'): # NOTE: 大约5万5千步1个epoch
        super(Model_behavior_clone, self).__init__()
        self.bert = bert_common.initiate_bert()
        self.train_step_counter = 4990
        self.loss_step_interval = loss_step_interval
        self.valid_step_interval = valid_step_interval
        self.prefix = prefix
        self.max_valid_score = -1
        self.last_valid_score = -1
        self.logger = common.Logger_simple(self.prefix)
        self.losses = []
        self.temp_loss_accumurated = 0
        self.valid_scores = []
    def init_optimizer(self): # TESTING: 使用accelarate库
        self.cuda()
        self.train()
        from accelerate import Accelerator
        accelerator = Accelerator()
        self.device = accelerator.device
        optimizer = optim.AdamW(self.parameters(), lr=2e-5) # 从1e-3到2e-5
        self, optimizer = accelerator.prepare(
            self, optimizer
        )
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.prepared_for_training = True
    def draw_line_chart(self):
        draw_line_chart(list(range(len(self.losses))), [self.losses, self.valid_scores], ['loss', 'valid_scores'], path = f'exp/auto_filename/{self.prefix}.png')
    def train(self):
        self.bert.train()
    def eval(self):
        self.bert.eval()
    def backward_may_step(self, loss):
        self.accelerator.backward(loss)
        self.temp_loss_accumurated += loss.item()
        self.train_step_counter += 1
        if self.train_step_counter % self.loss_step_interval == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.losses.append(min(self.temp_loss_accumurated / self.loss_step_interval * 0.1, 1.0))
            self.valid_scores.append(self.last_valid_score)
            self.temp_loss_accumurated = 0
        print(f'train_step_counter = {self.train_step_counter}')
    def may_save_checkpoint(self):
        if self.train_step_counter % self.valid_step_interval == 0:
            now_valid_score = self.valid() # will switch back to train mode
            self.last_valid_score = now_valid_score
            if now_valid_score > self.max_valid_score:
                self.max_valid_score = now_valid_score
                self.logger.add(f'Checkpoint saved! valid_score = {now_valid_score}')
                self.logger.write_txt_log()
                self.draw_line_chart()
                self.save_checkpoint()
    def save_checkpoint(self):
        self.bert.save_pretrained(f'exp/auto_filename/{self.prefix}.tch')
    def load_checkpoint(self):
        self.bert, _ = bert_common.load_trained_model(f'exp/auto_filename/{self.prefix}.tch')
        self.bert.cuda()
    def valid(self):
        from bert_common import batch_valid
        self.eval()
        valid_score = batch_valid(self.bert, save_readable=False, steps_limit = 50)
        self.train()
        self.logger.add(f'train_step_counter = {self.train_step_counter}, valid_score = {valid_score}')
        self.logger.write_txt_log()
        return valid_score
    def test_full(self):
        from bert_common import run_test_full
        run_test_full(self.bert, file_prefix=self.prefix)



def train_loop_neo(model: Model_behavior_clone, game_paths, epoch = 1):
    model.train()
    for e in range(epoch):
        for game_path in game_paths:
            game = Game_for_rl(game_path)
            walkthroughs = game.filter_walkthroughs_with_input_test()
            game = Game_for_rl(game_path)
            game.reset()
            for command in walkthroughs:
                state = game.get_state()
                x = state.x
                y = state.action_list.index(command)
                loss = get_loss(model.bert, x, y, model.device)
                model.backward_may_step(loss)
                model.may_save_checkpoint()
                game.input(command)

@lru_cache(maxsize=None)
def my_random_train_paths():
    # prepare dataset
    from ftwp_info import all_train_game_paths
    game_paths = all_train_game_paths()
    import random
    random.seed(2025)
    random.shuffle(game_paths)
    return game_paths

def train(repeat = 3, epoch = 4):
    game_paths = my_random_train_paths()
    # train
    for i in range(repeat):
        model = Model_behavior_clone(prefix=f'bert_behavior_clone{i}')
        model.init_optimizer()
        # model.may_save_checkpoint() # NOTE: valid and save checkpoint before training
        train_loop_neo(model, game_paths, epoch = epoch)

def test_model_after_training(repeat = 3):
    for i in range(repeat):
        model = Model_behavior_clone(prefix=f'bert_behavior_clone{i}')
        model.load_checkpoint()
        model.eval()
        model.test_full()

def night_run():
    RP = 3
    train(repeat = RP, epoch = 4)
    test_model_after_training(repeat=RP)
    common.shutdown()


# ======================= 训练 =============================

# 实验结果已经出来了, 查看docs/ftwp_bert_baseline，epoch设定为2即可
def early_stop_exp():
    train(epoch = 4)


# @hitory: 2025.1.20 将epoch从1更改为2
def train_several_models(count = 3):
    for i in range(count):
        filename = f'baseline_restart{i}'
        _ = train(epoch=2)


def exp_and_shutdown():
    try:
        # early_stop_exp()
        train_several_models(count = 3)
    except Exception as ex:
        f = open('exp/auto_filename/error.txt', 'w')
        f.write(str(ex))
        f.close()
    common.shutdown()