import logging
import torch
from nltk import word_tokenize
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForTokenClassification

MODEL = None

BERT_BASE_UNCASED_MODEL_ID = "bert-base-uncased"
DEBUG = False
if DEBUG:
    logging.basicConfig(filename='gameplay.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger('agent')
dbg = logger.debug

import re

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


def load_trained_model(path):
    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)
    model = BertForTokenClassification.from_pretrained(path)
    _ = model.cuda()
    return model, tokenizer

class Ner:

    def __init__(self, model_dir: str = '/home/taku/Downloads/cog2019_ftwp/trained_models/bert_ner/bert_ner.tch', device: str = 'cuda'):
        self.device = device
        self.model, self.tokenizer = load_trained_model(model_dir)
        self.max_seq_length = 128
        self.model.eval()

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        ## insert "[SEP]"
        tokens.append("[SEP]")
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        input_ids,input_mask,segment_ids,valid_positions = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long)
        input_mask = torch.tensor([input_mask],dtype=torch.long)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask = input_mask).logits
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label[0])]

        logits_label = [logits_label[0][index] for index,i in enumerate(input_mask[0]) if i.item()==1]
        logits_label.pop(0)
        logits_label.pop()

        assert len(logits_label) == len(valid_positions)
        labels = []
        for valid,label in zip(valid_positions,logits_label):
            if valid:
                labels.append(self.label_map[label])
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word": word, "tag":label,"confidence":confidence}
                 for word,label,confidence in zip(words,labels,logits_confidence)]
        return output

    def save_checkpoint(self):
        self.model.save_pretrained(f'exp/auto_filename/{self.prefix}.tch')
    def load_checkpoint(self, path = None):
        if not path:
            self.model, _ = load_trained_model(f'exp/auto_filename/{self.prefix}.tch')
        else:
            self.model, _ = load_trained_model(path)


def extract_entity_list(predictions):
    res = []
    in_progress = ''
    etyp = ''
    for word in predictions:
        if word['tag'] == 'O':
            if in_progress:
                res.append((in_progress.strip(), etyp))
                in_progress, etyp = '', ''
        elif word['tag'].startswith('B-'):
            if in_progress:
                res.append((in_progress.strip(), etyp))
            in_progress = word['word']
            etyp = word['tag'][2:]
        elif word['tag'].startswith('I-'):
            in_progress += ' ' + word['word']
            etyp = word['tag'][2:]
    if in_progress:
        res.append((in_progress.strip(), etyp))
    return list(set(res))

# 通过环境描述和库存信息提取实体
def extract_entities(description, inventory, model=None): 
    global MODEL
    if model is None:
        if MODEL is None:
            modelpath = '/home/taku/Downloads/cog2019_ftwp/trained_models/bert_ner/bert_ner.tch'
            MODEL = Ner(modelpath)
        model = MODEL

    cp = CompactPreprocessor()
    state = cp.convert(description, '', inventory, [])

    pred = model.predict(state.lower())
    entities_types = extract_entity_list(pred)
    return entities_types


class CommandGenerator:
    def __init__(self):
        self.rgx = re.compile(r'\b(\w+\-\w+)\b')
        self.hifen_map = {}
        self.hifen_rev_map = {}
        self.langmodel = self.default_template_mapper()

    def entities_mapping(self, entities):
        res = []
        for e,t in entities:
            for k in self.hifen_rev_map.keys():
                if k in e:
                    e = e.replace(k, self.hifen_rev_map[k])
            res.append((e,t))
        return res

    def generate_entities(self, description, inventory):
        entities = extract_entities(description, inventory, model=self.ner) # 通过环境描述和库存信息提取实体
        return self.entities_mapping(entities)


    def preprocess_description(self, description):
        mobj = self.rgx.search(description)
        if mobj:
            kw = mobj.group(0) # 应该是匹配到的第一个带有-的词 a-b
            target = kw.replace('-', ' ') # a b
            self.hifen_map[kw] = target
            self.hifen_rev_map[target] = kw
            return description.replace(kw, target)
        return description

    def commands_generate(self, infos):
        description = self.preprocess_description(infos['description'])
        inventory = infos['inventory']
        entity_types = self.generate_entities(description, inventory) # 通过ner模型，提取实体
        # entities = [e for e,_ in entity_types] 
        templates = infos['command_templates']
        commands = self.generate_commands_by_template(description, templates, entity_types) # 通过实体和菜单生成指令，为什么要菜单？
        return commands

    def generate_commands_by_template(self, description, command_templates, entities):
        # description = self.preprocess_description(infos['description'])
        commands = self.get_admissible_commands(description, entities, command_templates) # Q: command_templates怎么来的？
        return commands


    def default_template_mapper():
        from template_mapper import CommandModel
        langmodel = CommandModel()
        return langmodel

    # NOTE: 这里没用到inventory，但是上面的genrate_entities用到了
    def get_admissible_commands(self, description, entities, templates): 
        state_entities = entities
        dbg('State entities: {}'.format(sorted(state_entities)))
        cmds = self.langmodel.generate_all(state_entities, templates) # 通过langmodel和识别出来的实体生成所有可能的指令
        if 'cookbook' in description and 'examine cookbook' not in cmds:
            cmds.append('examine cookbook')
        return cmds