import os
import pandas as pd
from tqdm import tqdm
import re
import random
from qamodel import QAModel, default_tokenizer
import qamodel
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
                             SequentialSampler
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
import logging
from functools import lru_cache

DEBUG = True
if DEBUG:
    logging.basicConfig(filename='../exp/auto_filename/qa_model_train.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger('ner_command_generator')
dbg = logger.debug

class CompactPreprocessor:
    """ Preprocessor that tries to reduce the number of tokens by removing punctuation
    """
    def convert(self, description, recipe, inventory, entities):
        if not recipe or isinstance(recipe, float): # catch NaN
            recipe = 'missing recipe'
        txt = self.inventory_text(inventory, entities) + ' ' + recipe + ' ' + description
        txt = re.sub(r'\n', ' ', txt)
        # convert names with hiffen with space
        txt = re.sub(r'(\w)\-(\w)', r'\1 \2', txt)
        # remove punctuation
        txt = re.sub(r'([.:\-!=#",?])', r' ', txt)
        txt = re.sub(r'\s{2,}', ' ', txt)
        return txt.strip('.')
    
    def convert_datarow(self, row):
        description = row['description']
        recipe = row['recipe']
        inventory = row['inventory']
        entities = row['entities']
        return self.convert(description, recipe, inventory, entities)

    def inventory_text(self, inventory, entities):
        n_items = self.count_inventory_items(inventory, entities)
        text = '{} {}'.format(n_items, inventory)
        return text

    def count_inventory_items(self, inventory, entities):
        parts = [p.strip() for p in inventory.split('\n')]
        parts = [p for p in parts if p]
        return len([p for p in parts if any(p.find(ent) != -1 for ent in entities)])

def make_dataset(df, sample=5, sample2=0):
    irrelevant_commands = ['close', 'drink', 'insert', 'put']
    data = []
    cp = CompactPreprocessor()
    # gen_commands是排列组合生成的所有可能的命令
    # taku NOTE: 注意这里将gen_commands替换成了admissible_commands，并且将sample2改为0（不会从不相关的指令中进行取样，admissible_commands中本来就没有不相关指令）
    cols = ['entities','inventory','recipe','description','command', 'gamename', 'admissible_commands']
    for entities, inventory, recipe, description, command, gamename, gen_commands in tqdm(df[cols].values):
        # add correct command for state
        # FIXME
        #entities = [e for e, _ in row.ext_entities]
        entities = [e.replace('-', ' ') for e in entities]

        text = cp.convert(description, recipe, inventory, entities)
        data.append((text, command, 1, gamename))
        # important commands
        main_commands = [c for c in gen_commands
                         if c != command and c not in irrelevant_commands]
        if sample > 0:
            main_commands = random.sample(main_commands, min(sample, len(main_commands)))
        # add wrong commands for state
        for cmd in main_commands:
            data.append((text, cmd, 0, gamename))
        # irrelevant commands
        bad_commands = [c for c in gen_commands
                         if c != command and c in irrelevant_commands]
        if sample2 > 0:
            bad_commands = random.sample(bad_commands, min(sample2, len(bad_commands)))
        # add wrong commands for state
        for cmd in bad_commands:
            data.append((text, cmd, 0, gamename))

    return pd.DataFrame(data, columns=['text','command', 'target', 'gamename'])

def generate_qa_datasets(traindata):
    """ generate a dataset for QA model training """

    train = pd.read_csv(traindata)
    for c in ['entities', 'admissible_commands']:
        train[c] = train[c].apply(eval)

    qa_train = make_dataset(train, sample=5, sample2=0)
    return qa_train

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

def convert_examples_to_features(texts, labels, max_seq_length, tokenizer,
                                 label_map, texts_b=None): # texts_b不可能为None，如果为None，学习到个毛线
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
    for example, example_b, label in zip(texts, texts_b, labels): # example_b就是command
        tokens_a = tokenizer.tokenize(example) # 这里单纯是prompt

        tokens_b = None
        if has_text_b:
            tokens_b = tokenizer.tokenize(example_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3) # 因为tokens_b是单纯的command，不可能比a更长，所以只是在调整a的长度
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
        for token in tokens_a: # token in text
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
        features.append([input_ids, input_mask, segment_ids, label_id])
    return features

@lru_cache(maxsize=10)
def qa_dataloader(dataset_path = '/home/taku/Downloads/cog2019_ftwp/cogni_dataset', train = True, batch_size = 8):
    csv_name = 'walkthrough_train.csv' if train else 'walkthrough_valid.csv'
    df = os.path.join(dataset_path, csv_name)
    spawned_df = generate_qa_datasets(df) # 一个正确case，5个错误case
    max_seq_length = 480 # Added by taku
    label_map = {0:0, 1:1}
    tokenizer = default_tokenizer()
    features = convert_examples_to_features(spawned_df.text.values, spawned_df.target.values,
                                    max_seq_length, tokenizer, label_map,
                                    texts_b=spawned_df.command.values) # 这里的texts_b是所有的command
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[3] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def train(model = None, checkpoint_directory = '../exp/auto_filename', epoch = 0, batch_size = 16, suffix = ''):
    from accelerate import Accelerator
    accelerator = Accelerator()
    if not model:
        model = QAModel()
    model.cuda()
    model.train()
    train_dataloader = qa_dataloader(train = True, batch_size = batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    DEVICE = accelerator.device
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    global_step = 0
    predictions = np.array([])
    labels = np.array([])

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits, loss = model(input_ids, segment_ids, input_mask, label_ids)
        accelerator.backward(loss)
        # loss.backward() # NOTE: 使用accelarate加速
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        predictions = np.concatenate([predictions, np.argmax(logits, axis=1)])
        labels = np.concatenate([labels, label_ids])
        if global_step % 1000 == 0: # 每1000个step输出一次，这个值是过往的累计值
            acc = metrics.accuracy_score(labels, predictions)
            f1 = metrics.f1_score(labels, predictions)
            print('[{}] Loss {:f} Acc {:f} F1 {:f}'.format(step,
                        tr_loss/nb_tr_steps, acc, f1))
    # Save checkpoint each epoch
    checkpoint_path = os.path.join(checkpoint_directory, f'qa_model_checkpoint_epoch_{epoch}_{suffix}.tch')
    model.save_checkpoint(checkpoint_path)

# 这个函数得到的分数是命中率，而不是游戏上的分数
def validate(model, batch_size = 16):
    num_labels = 2
    # dummy label map
    DEVICE = 'cuda'
    valid_dataloader = qa_dataloader(train = False, batch_size = batch_size)
    model.cuda()
    model.eval()
    outputs = np.empty((0, num_labels))
    labels = np.array([])
    cumloss, nsteps = 0, 0
    for step, batch in enumerate(tqdm(valid_dataloader)):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            logits, loss = model(input_ids, segment_ids, input_mask, label_ids)
            proba = F.softmax(logits, dim=1)
            cumloss += loss.item()
        nsteps += 1
        outputs = np.vstack([outputs, proba.detach().cpu().numpy()])
        labels = np.concatenate([labels, label_ids.detach().cpu().numpy()])

    acc = metrics.accuracy_score(labels, np.argmax(outputs, axis=1))
    f1 = metrics.f1_score(labels, np.argmax(outputs, axis=1))
    dbg('Validation loss {:f} Acc {:f} F1 {:f}'.format(cumloss/nsteps, acc, f1))


def run_train(epochs = 5):
    model = QAModel()
    for epoch in range(epochs):
        train(model, epoch = epoch, batch_size=16, suffix=f'repeat_1')
        validate(model, batch_size=16)
