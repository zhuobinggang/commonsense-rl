import torch
from torch.utils.data import Dataset



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
    

def get_loss(model, tokenizer, x, y):
    inputs = tokenizer(x, return_tensors="pt")
    # logits = model(**inputs).logits
    # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    labels = x.replace('[MASK]', str(y.item()))
    labels = tokenizer(labels, return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(**inputs, labels=labels)
    return outputs.loss

def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), 2e-5)


class Logger:
    def __init__(self,batch_size = 4, log_step = 36):
        self.batch_size = batch_size
        self.losses = []
        self.temp_losses = []
        self.counter = 0
        self.log_step = log_step
    def add(self, loss):
        self.temp_losses.append(loss)
        self.counter += 1
        if self.counter >= self.batch_size:
            self.counter = 0
            self.losses.append(sum(self.temp_losses) / len(self.temp_losses))
            self.temp_losses = []
            if len(self.losses) % self.log_step == 0:
                self.draw()
    def get_losses(self):
        return self.losses
    def draw(self):
        # draw
        datas = self.get_losses()
        draw_line_chart(list(range(len(datas))), [datas], ['losses'])
    

def train_loop(model, tokenizer, batch = 4):
    # Accelerator
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    optimizer = get_optimizer(model)
    dataloader = get_dataloader(batch=batch)
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    logger = Logger(batch_size=batch)
    for xs, ys in next(iter(dataloader)):
        optimizer.zero_grad()
        for x,y in zip(xs, ys):
            loss = get_loss(model, tokenizer, x, y)
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


def train():
    from transformers import AutoTokenizer, ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBertForMaskedLM.from_pretrained(model_id)
    train_loop(model, tokenizer)
    return model, tokenizer