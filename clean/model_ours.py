import logging
from transformers import BertForMaskedLM, AutoTokenizer
from functools import lru_cache
import common_new as common
from common_new import draw_line_chart
import torch
import torch.optim as optim
from torch import nn
import re
import game
from dataset_create_taku import read_csv_dataset
from bert_utils import default_tokenizer, init_bert_ours
from bert_utils import get_next_command, tokenize_game_state, command_indexs_tokenized
from dataset_create_taku import row_to_game_state

DEBUG = True
if DEBUG:
    logging.basicConfig(filename='log/model_ours.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('cogni_agent_base')
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
    def loss_from_state(self, state:game.Game_state, action_idx: int):
        prompt_ids = tokenize_game_state(state)
        label_code_in_tokenizer = command_indexs_tokenized()[action_idx]
        labels = torch.empty(prompt_ids.shape, dtype=torch.long).fill_(-100)
        batch_idx, cls_idx = 0, 0
        labels[batch_idx, cls_idx] = label_code_in_tokenizer
        outputs = self.bert(input_ids=prompt_ids, labels=labels)
        return outputs.loss
    def predict(self, game_state:game.Game_state):
        return get_next_command(self.bert, game_state)
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
    g = game.default_game()
    _ = g.reset()
    g.act('go east')
    gs = game.game_state_from_game(g)
    # loss = m.loss_from_info(gs, 11)
    # print(loss)
    # print(m.predict(gs))
    return m, g, gs

def test2():
    m = Model()
    g = game.default_game()
    _ = g.reset()
    g.act('go east')
    state = game.game_state_from_game(g)
    prompt_ids = tokenize_game_state(state)
    action_idx = 11
    label_code_in_tokenizer = command_indexs_tokenized()[action_idx]
    return m, g, state

# TODO: 使用accerlerate
def train():
    from accelerate import Accelerator
    accelerator = Accelerator()
    model = Model()
    model.init_bert()
    optimizer = optim.AdamW(m.parameters(), lr=2e-5) # 从1e-3到2e-5
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    csv = read_csv_dataset(split='train')
    for _, row in csv.iterrows():
        state = row_to_game_state(row)
        action_idx = state.filtered_available_commands().index(row['action'])
        loss = model.loss_from_state(state, action_idx)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()