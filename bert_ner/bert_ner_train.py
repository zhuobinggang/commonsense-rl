
import os
import logging
from typing import List
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import torch.nn.functional as F
from seqeval.metrics import classification_report
import torch.optim as optim
from transformers import AutoTokenizer, BertForTokenClassification
from functools import lru_cache

BERT_BASE_UNCASED_MODEL_ID = "bert-base-uncased"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

 
@lru_cache(maxsize=32)
def default_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)

def default_model(num_labels):
    return BertForTokenClassification.from_pretrained(BERT_BASE_UNCASED_MODEL_ID,
                num_labels = num_labels)

def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


def get_labels():
    #return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
    return ["O", "B-D", "I-D",  "B-C", "I-C", "B-W", "I-W", "B-S", "I-S", "B-T", "I-T", "B-F", "I-F",
            "X", "[CLS]", "[SEP]"]

class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        #return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        return ["O", "B-D", "I-D",  "B-C", "I-C", "B-W", "I-W", "B-S", "I-S", "B-T", "I-T", "B-F", "I-F",
                "X", "[CLS]", "[SEP]"]

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples
    
def convert_examples_to_features(examples: List[InputExample], label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features


# DONE: 证实是以一个step的inventory + 环境描述为单位进行ner
def get_train_examples():
    processor = NerProcessor()
    train_examples = processor.get_train_examples('/home/taku/Downloads/cog2019_ftwp/games/cogit_agent_ner_dataset/nerdata/') # 读取train.txt
    return train_examples

def train_data_prepare():
    train_examples = get_train_examples()
    label_list = get_labels()
    max_seq_length = 128 # 为什么只能128? 感觉完全可以用更长的序列
    train_batch_size = 8
    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer = default_tokenizer())
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    return train_dataloader

def train_loop_old(model, optimizer, train_dataloader, gradient_accumulation_steps = 1, num_train_epochs = 1):
    # 上面还是在做数据处理，下面才是训练
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

# TODO: 用accelerate加速训练
def train_loop(model, gradient_accumulation_steps = 1, num_train_epochs = 1):
    global_step = 0
    from accelerate import Accelerator
    accelerator = Accelerator()
    # 上面还是在做数据处理，下面才是训练
    model.train()
    device = accelerator.device
    train_dataloader = train_data_prepare()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) # 从1e-3到2e-5
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # loss = model(input_ids, segment_ids, input_mask, label_ids).loss
            loss = model(input_ids, attention_mask = input_mask, labels = label_ids).loss

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # loss.backward()
            accelerator.backward(loss)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

def valid(model, tokenizer, data_dir = '/home/taku/Downloads/cog2019_ftwp/games/cogit_agent_ner_dataset/nerdata/'):
    output_dir = 'exp/auto_filename'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = NerProcessor()
    eval_examples = processor.get_dev_examples(data_dir)
    label_list = get_labels()
    max_seq_length = 128
    eval_batch_size = 8
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    label_map = {i : label for i, label in enumerate(label_list,1)} # {1: 'O', 2: 'I-D' ...}
    label_map[0] = 'Error'
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask = input_mask).logits
        
        # logger.info('XXXXXXXXXXXXXXXXXXXX')
        # logger.info(logits.shape) # 8, 128, 17
        # logger.info(logits)

        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2) # 8, 128
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        for i,mask in enumerate(input_mask): # i 是batch
            temp_1 =  []
            temp_2 = []
            for j, m in enumerate(mask): # j = 128, m = 1 or 0
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids[i][j]] != "X":
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
                else:
                    temp_1.pop()
                    temp_2.pop()
                    break
            y_true.append(temp_1)
            y_pred.append(temp_2)
    report = classification_report(y_true, y_pred,digits=4) # TODO： 观察
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("\n%s", report)
        writer.write(report)

def load_trained_model(path):
    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)
    model = BertForTokenClassification.from_pretrained(path)
    _ = model.cuda()
    return model, tokenizer

class Tester:
    def __init__(self):
        label_list = get_labels()
        num_labels = len(label_list) + 1
        self.bert = default_model(num_labels)
        self.num_labels = num_labels
        self.tokenizer = default_tokenizer()
        self.prefix = 'bert_ner'
        
    def run_train(self):
        train_loop(self.bert, num_train_epochs = 2)
        self.save_checkpoint()
    
    def run_eval(self):
        valid(self.bert, self.tokenizer)

    def save_checkpoint(self):
        self.bert.save_pretrained(f'exp/auto_filename/{self.prefix}.tch')
    def load_checkpoint(self, path = None):
        if not path:
            self.bert, _ = load_trained_model(f'exp/auto_filename/{self.prefix}.tch')
        else:
            self.bert, _ = load_trained_model(path)