import torch
from torch import nn
from transformers import BertModel
from transformers import AutoTokenizer
# from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from functools import lru_cache
import re
import numpy as np
import logging

DEBUG = True
if DEBUG:
    logging.basicConfig(filename='exp/auto_filename/cogni_agent_logger.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('cogni_agent_base')
dbg = logger.debug

BERT_BASE_UNCASED_MODEL_ID = 'bert-base-uncased'

@lru_cache(maxsize=1)
def default_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)

def init_bert():
    model = BertModel.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)
    return model

def load_trained_model(path):
    tokenizer = default_tokenizer()
    model = BertModel.from_pretrained(path)
    model.cuda()
    return model, tokenizer

class Game_state:
    def __init__(self) -> None:
        self.x = ''
        self.action_list = []
        # 2025.3.16 散装的信息
        self.location = '' # room name
        self.recipe_raw = '' # recipe text
        self.recipe = '' # recipe text
        self.inventory_raw = '' # inventory item string
        self.inventory = '' # inventory item string set
        self.action_obs_pairs = [] # (action, obs)
        self.world_map = {} # 用于存储地图
        self.description = '' # 用于存储游戏描述
        self.command_templates = [] # 用于存储command_templates
        self.entities = []


class QAModel(nn.Module):
    def __init__(self, num_labels = 2):
        super(QAModel, self).__init__()
        self.num_labels = num_labels
        self.bert = init_bert()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.checkpoint_name = 'qa_model'
        self.max_seq_length = 480
        self.tokenizer = default_tokenizer()
        self.eval_batch_size = 16
        self.device = 'cuda'
        self.cuda()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        out = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        pooled_output = out.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # (batch_size, num_labels)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss
        else:
            return logits

    def save_checkpoint(self, path = None):
        if not path:
            self.bert.save_pretrained(f'../exp/auto_filename/{self.checkpoint_name}.tch')
        else:
            self.bert.save_pretrained(path)

    def load_checkpoint(self, path = None):
        if not path:
            self.bert, _ = load_trained_model(f'../exp/auto_filename/{self.checkpoint_name}.tch')
        else:
            self.bert, _ = load_trained_model(path)
        self.bert.cuda()

    def get_command_probs(self, game_state: Game_state):
        # 将game_state的信息压缩
        pp = CompactPreprocessor()
        prompt = pp.convert(game_state.description, game_state.recipe_raw, game_state.inventory_raw, game_state.entities)
        commands = game_state.action_list
        # 将所有commands装入一个Dataloader中，同时处理
        evaldata = bprocess([prompt]*len(commands),
                            self.max_seq_length, commands, self.tokenizer)
        sampler = SequentialSampler(evaldata)
        dataloader = DataLoader(evaldata, sampler=sampler,
                                batch_size=self.eval_batch_size) # 将所有commands装入一个Dataloader中，同时处理
        probs = []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch

            with torch.no_grad():
                pred = self.forward(input_ids, segment_ids, input_mask) # (batch_size, 2)
                pred = pred[:,1].detach().cpu().numpy() # (batch_size) 相当于模型对于这个command的置信度
                probs.append(pred)
                # self.n_model_evals += 1
        probs = np.concatenate(probs) if len(probs) > 0 else np.array(probs) # (commands_length)

        top_commands = list(reversed(sorted(zip(commands, probs), key=lambda x: x[1]))) # 根据置信度排序
        commands = [v[0] for v in top_commands]
        probs = np.array([v[1] for v in top_commands])
        # dbg('State: {}'.format(prompt))
        # dbg('Predicted: ' + ','.join(
        #         '({}, {:5.2f})'.format(c,p) for c,p in reversed(
        #             sorted(zip(commands, probs), key=lambda x: x[1]))))
        return commands, probs
    
    def next_action(self, game_state):
        commands, probs = self.get_command_probs(game_state)
        return commands[0]
    
    def test_full(self):
        from bert_common import run_test_full
        run_test_full(self, file_prefix=self.checkpoint_name)

    def test_partial(self, game_count = 30):
        from bert_common import run_test_full
        run_test_full(self, file_prefix=self.checkpoint_name, game_count=game_count)



class CompactPreprocessor:
    """ Preprocessor that tries to reduce the number of tokens by removing punctuation
    """
    def convert(self, description, recipe, inventory, entities):
        if recipe != '':
            txt = self.inventory_text(inventory, entities) + ' ' + recipe + ' ' + description
        else:
            txt = self.inventory_text(inventory, entities) + ' missing recipe ' + description
        txt = re.sub(r'\n', ' ', txt)
        # convert names with hiffen with space
        txt = re.sub(r'(\w)\-(\w)', r'\1 \2', txt)
        # remove punctuation
        txt = re.sub(r'([.:\-!=#",?])', r' ', txt)
        txt = re.sub(r'\s{2,}', ' ', txt)
        return txt.strip('.')

    def inventory_text(self, inventory, entities):
        n_items = self.count_inventory_items(inventory, entities)
        text = '{} {}'.format(n_items, inventory)
        return text

    def count_inventory_items(self, inventory, entities):
        parts = [p.strip() for p in inventory.split('\n')]
        parts = [p for p in parts if p]
        return len([p for p in parts if any(p.find(ent) != -1 for ent in entities)])

def text_preprocess(state, entities):
    pp = CompactPreprocessor()
    return pp.convert(state.description, state.recipe, state.inventory, entities)


# 提供的texts_b是commands，对应的segment_ids是1
def bprocess(texts, max_seq_length, texts_b=None, tokenizer=None):
    dummy_target = [0]*len(texts)
    dummy_label_map = {0:0 for _ in range(len(texts))}
    features = convert_examples_to_features(texts, dummy_target,
                                    max_seq_length, tokenizer, dummy_label_map,
                                    texts_b=texts_b)
    input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)

    return TensorDataset(input_ids, input_mask, segment_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(texts, labels, max_seq_length, tokenizer, label_map, texts_b=None):
    """
    Loads a data file into a list of `InputBatch`s.
    (adapted from pytorch_pretrained_bert run_classifier.py)
    """

    assert len(texts) == len(labels)
    if texts_b is None:
        # dummy texts
        texts_b = ['' for _ in range(len(texts))]
        has_text_b = False
    else:
        assert len(texts_b) == len(labels)
        has_text_b = True

    features = []
    for example, example_b, label in zip(texts, texts_b, labels):
        tokens_a = tokenizer.tokenize(example)

        tokens_b = None
        if has_text_b:
            tokens_b = tokenizer.tokenize(example_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label]
        features.append([input_ids,input_mask,segment_ids,label_id])
    return features
