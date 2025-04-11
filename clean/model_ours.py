from transformers import BertForMaskedLM, AutoTokenizer
from functools import lru_cache
import common_new as common
from common_new import draw_line_chart
import torch
import torch.optim as optim
from torch import nn
import re
from game import Game_handle_recipe, game_state_from_game, Game_state, default_game, test_game
from dataset_create_taku import read_csv_dataset, get_cv_games
from bert_utils import default_tokenizer, init_bert_ours, action_select_loss, action_select_loss_batched
from bert_utils import get_next_command, tokenize_game_state, command_indexs_tokenized, to_bert_input, DEVICE
from dataset_create_taku import row_to_game_state
from tqdm import tqdm
from recordclass import recordclass
import numpy as np
from typing import List
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

Statistic = recordclass('Statistic', 'losses')

import logging
logger = logging.getLogger('model_ours')
dbg = logger.debug


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.bert = init_bert_ours()
        self.bert = None
        self.prefix = 'bert_ours'
    def init_bert(self):
        if not self.bert:
            self.bert = init_bert_ours()
    def loss_from_state(self, state:Game_state, action_idx: int):
        return action_select_loss(self.bert, state, action_idx)
    def loss_from_state_batched(self, states:List[Game_state], action_idxs:List[int]):
        return action_select_loss_batched(self.bert, states, action_idxs)
    def predict(self, game_state:Game_state):
        result = get_next_command(self.bert, game_state)
        return result.command
    def save_checkpoint(self, base_path = 'log', epoch = -1):
        path = f'{base_path}/{self.prefix}_epoch_{epoch}.pth'
        torch.save({
            'iteration': epoch,
            'state': self.state_dict(),
        }, path)
    def load_checkpoint(self, path):
        self.init_bert() # NOTE: 需要先初始化然后加载
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['state'])

def test():
    m = Model()
    g = default_game()
    _ = g.reset()
    g.act('go east')
    gs = game_state_from_game(g)
    # loss = m.loss_from_info(gs, 11)
    # print(loss)
    # print(m.predict(gs))
    return m, g, gs

def test2():
    m = Model()
    g = default_game()
    _ = g.reset()
    g.act('go east')
    state = game_state_from_game(g)
    prompt_ids = tokenize_game_state(state)
    action_idx = 11
    label_code_in_tokenizer = command_indexs_tokenized()[action_idx]
    return m, g, state

@lru_cache(maxsize=1)
def dataloader_get(split = 'train', batch_size = 8):
    # dataloader
    csv = read_csv_dataset(split=split)
    csv = csv.sample(frac=1) # shuffle to train
    bert_inputs = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row)
        action_idx = state.filtered_available_commands().index(row['action'])
        bert_input = to_bert_input(state, action_idx)
        bert_inputs.append(bert_input)
    all_input_ids = torch.tensor([bert_input.input_ids for bert_input in bert_inputs], dtype=torch.long)
    all_attention_mask = torch.tensor([bert_input.attention_mask for bert_input in bert_inputs], dtype=torch.long)
    all_label_ids = torch.tensor([bert_input.labels for bert_input in bert_inputs], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

@lru_cache(maxsize=1)
def get_writer():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    return writer

def train(model, batch_size = 8, split = 'train'):
    train_dataloader = dataloader_get(split=split, batch_size=batch_size)
    # training
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    model.train()
    dbg('Model train on.')
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) # 从1e-3到2e-5
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, input_mask, label_ids = batch
        outputs = model.bert(input_ids=input_ids.to(DEVICE), 
                   attention_mask=input_mask.to(DEVICE), 
                   labels=label_ids.to(DEVICE))
        loss = outputs.loss
        accelerator.backward(loss)
        get_writer().add_scalar('Loss/train', loss.item(), batch_idx)
        optimizer.step()
        optimizer.zero_grad()

def valid_all(model: Model, split = 'partial_valid'):
    game_paths = get_cv_games(split=split)
    score = 0
    max_score = 0
    dbg(f'Validating {split} games, total {len(game_paths)}')
    for game_path in tqdm(game_paths, desc=f"Validating {split} games"):
        game = Game_handle_recipe(game_path)
        result = test_game(game, model)
        score += result.score
        max_score += result.max_score
        dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    return score / max_score

def valid_all_by_model_path(model_path: str):
    model = Model()
    model.load_checkpoint(model_path)
    return valid_all(model)

# ================================

def get_model(checkpoint_path = None):
    model = Model()
    model.prefix = 'roberta_ours'
    model.init_bert()
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    return model

def train_reapeat(repeat = 3, epoch = 3, batch_size = 8):
    for rp in range(repeat):
        model = get_model()
        model.prefix = f'roberta_ours_repeat_{rp}'
        for i in range(epoch):
            train(model, batch_size=batch_size, split='train')
            score = valid_all(model, split='partial_valid')
            dbg(f'Valid results, repeat {rp} epoch {i} score: {score}')
            # get_writer().add_scalar(f'Score/valid_rp{rp}', score, i)
            model.save_checkpoint(base_path= '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours', epoch=i)

# 补充2个epoch的训练
def train_reapeat_plus(batch_size = 8):
    start = 2
    for rp in range(3):
        path = f'/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours/roberta_ours_repeat_2_epoch_{start}.pth'
        model = get_model(path)
        model.prefix = f'roberta_ours_repeat_{rp}'
        for epoch_continue in range(3, 5): # epoch 3 ~ 5
            train(model, batch_size=batch_size, split='train')
            score = valid_all(model, split='partial_valid')
            dbg(f'Valid results, repeat {rp} epoch {epoch_continue} score: {score}')
            # get_writer().add_scalar(f'Score/valid_rp{rp}', score, i)
            model.save_checkpoint(base_path= '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours', epoch=epoch_continue)

def test_trained():
    best_model_index = [1,2,1]
    for rp in range(3):
        model = get_model(f'/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours/roberta_ours_repeat_{rp}_epoch_{best_model_index[rp]}.pth')
        s1 = valid_all(model, split='valid')
        dbg(f'Full valid score ({rp}): {s1}')
        s2 = valid_all(model, split='test')
        dbg(f'Full test score ({rp}): {s2}')