import torch
from torch.utils.data import Dataset
import common


class CustomDataset(Dataset):
    def __init__(self):
        import pandas as pd
        self.dataframe = pd.read_json(path_or_buf='/home/taku/Downloads/cog2019_ftwp/procceeded_training_files/bert_trains.jsonl' ,lines=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.x.iloc[idx], self.dataframe.y.iloc[idx]
    
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
    

def get_loss(model, tokenizer, x, y , device = 'cpu'):
    inputs = tokenizer(x, return_tensors="pt")
    # logits = model(**inputs).logits
    # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    labels = x.replace('[MASK]', str(y.item()))
    labels = tokenizer(labels, return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(**inputs.to(device), labels=labels.to(device))
    return outputs.loss

def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), 2e-5)


class Logger:
    def __init__(self, model, batch_size = 4, log_step = 100, file_name = ''):
        self.batch_size = batch_size
        self.losses = []
        self.temp_losses = []
        self.counter = 0
        self.global_counter = 0
        self.update_checkpoint_timestep = 0
        self.log_step = log_step
        self.model = model
        self.valid_scores = []
        self.max_score = -1
        self.min_delta = 0.01 # 只有超过这个阈值的提升才被视作提升
        self.file_name = file_name or 'default'
        self.checkpoint_path = f'exp/auto_filename/{self.file_name}.tch'
        self.image_path = f'exp/auto_filename/{self.file_name}.png'
        self.text_log_path = f'exp/auto_filename/{self.file_name}.txt'
        self.text_log = ''
    def cal_valid_score_and_save_checkpoint(self):
        from bert_rl import batch_valid
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
    def last_valid_score(self):
        if len(self.valid_scores) > 0:
            return self.valid_scores[-1]
        else:
            return 0
    def save_checkpoint(self):
        print('CHECK POINT SAVED!')
        self.model.save_pretrained(self.checkpoint_path)
    def add(self, loss):
        self.temp_losses.append(loss)
        self.counter += 1
        self.global_counter += 1
        if self.counter >= self.batch_size:
            self.counter = 0
            self.losses.append(sum(self.temp_losses) / len(self.temp_losses))
            self.temp_losses = []
            if len(self.losses) % self.log_step == 0:
                self.valid_scores.append(self.cal_valid_score_and_save_checkpoint())
                self.draw()
                self.write_txt_log()
            else:
                self.valid_scores.append(self.last_valid_score())
    def get_losses(self):
        return self.losses
    def write_txt_log(self):
        self.text_log += f'{common.get_time_str()}\n'
        f = open(self.text_log_path, 'w')
        f.write(self.text_log)
        f.close()
    def draw(self):
        # draw
        losses = self.get_losses()
        draw_line_chart(list(range(len(losses))), [losses, self.valid_scores], ['losses', 'valid_scores'], path=self.image_path)

def train_loop(model, tokenizer, batch = 4, file_name = '', epoch = 1):
    # Accelerator
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    optimizer = get_optimizer(model)
    dataloader = get_dataloader(batch=batch)
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    logger = Logger(model, batch_size=batch, file_name=file_name)
    for e in range(epoch):
        for xs, ys in iter(dataloader):
            optimizer.zero_grad()
            for x,y in zip(xs, ys):
                loss = get_loss(model, tokenizer, x, y, device = device)
                logger.add(loss.item())
                accelerator.backward(loss)
            optimizer.step()


def draw_line_chart(x, ys, legends, path = 'exp/auto_filename/dd.png', colors = None, xlabel = None, ylabel = None):
    import matplotlib.pyplot as plt
    plt.clf()
    for i, (y, l) in enumerate(zip(ys, legends)):
        if colors is not None:
            plt.plot(x[:len(y)], y, colors[i], label = l)
        else:
            plt.plot(x[:len(y)], y, label = l)
    plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(path)


# ======================= 训练 =============================


def train(filename = '', epoch = 1):
    from transformers import AutoTokenizer, ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBertForMaskedLM.from_pretrained(model_id)
    model = model.train()
    train_loop(model, tokenizer, file_name=filename, epoch = epoch)
    return model, tokenizer

# 实验结果已经出来了, 查看docs/ftwp_bert_baseline，epoch设定为2即可
def early_stop_exp():
    train(filename='early_stop_exp', epoch = 3)


# @hitory: 2025.1.20 将epoch从1更改为2
def train_several_models(count = 3):
    for i in range(count):
        filename = f'baseline_restart{i}'
        _ = train(filename, epoch=2)


def exp_and_shutdown():
    try:
        # early_stop_exp()
        train_several_models(count = 3)
    except Exception as ex:
        f = open('exp/auto_filename/error.txt', 'w')
        f.write(str(ex))
        f.close()
    import os
    os.system('shutdown')