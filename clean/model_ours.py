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

DEBUG = True
if DEBUG:
    logging.basicConfig(filename='model_ours.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('cogni_agent_base')
dbg = logger.debug

BERT_BASE_UNCASED_MODEL_ID = 'bert-base-uncased'

@lru_cache(maxsize=1)
def default_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)

def init_bert_ours():
    return BertForMaskedLM.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = init_bert_ours()
    def loss_from_info(self, game_state:game.Game_state, action_idx):
        x = game.bert_prompt_from_game_state(game_state)
        tokenizer = default_tokenizer()
        prompt_ids = tokenizer('[CLS] ' + x + ' [SEP]', return_tensors="pt", add_special_tokens=False)["input_ids"] # [[token_length]]
        label_id = tokenizer(str(action_idx), return_tensors="pt", add_special_tokens=False)["input_ids"] # [idx]
        labels = torch.empty(prompt_ids.shape, dtype=torch.long).fill_(-100)
        labels[0, 0] = label_id[0, 0]
        outputs = self.bert(input_ids=prompt_ids, labels=labels)
        return outputs.loss
    def predict(self, game_state:game.Game_state):
        x = game.bert_prompt_from_game_state(game_state)
        tokenizer = default_tokenizer()
        prompt_ids = tokenizer('[CLS] ' + x + ' [SEP]', return_tensors="pt", add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            logits = self.bert(input_ids=prompt_ids).logits
        predicted_token_id = logits[0, 0].argmax(axis=-1)
        return tokenizer.decode(predicted_token_id)

def test():
    m = Model()
    g = game.default_game()
    _ = g.reset()
    gs = game.game_state_from_game(g)
    loss = m.loss_from_info(gs, 11)
    print(loss)
    print(m.predict(gs))
    return m, g, gs