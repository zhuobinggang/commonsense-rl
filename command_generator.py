import logging
import torch
from nltk import word_tokenize
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForTokenClassification
import re
import common
from tqdm import tqdm

MODEL = None

BERT_BASE_UNCASED_MODEL_ID = "bert-base-uncased"
DEBUG = True
if DEBUG:
    logging.basicConfig(filename='exp/auto_filename/ner_command_generator.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger('ner_command_generator')
dbg = logger.debug

class CommandModel:
    """ Generates commands based on command templates and entities """

    def __init__(self):
        self.template_cache = {}
        self.template_mapper = {
            '{d}': ['D', 'C'],
            '{f}': ['F'],
            '{s}': ['C', 'S'],
            '{o}': ['F', 'T'],
            '{x}': ['W']
        }

    def command_parser(self, cmd):
        """ parse the command into verb|entity|preposition|entity2 """
        mobj = re.search(r'([\w\-\{\} ]+) (in|with|on|into|from) ([\w\-\{\} ]+)', cmd)
        if mobj:
            base, preposition, entity2 = mobj.groups()
        else:
            base = cmd
            preposition, entity2 = '', ''

        parts = base.split()
        verb, entity = parts[0], ' '.join(parts[1:])
        return {'verb': verb, 'entity': entity, 'preposition': preposition,
                'entity2': entity2}

    def filter_templates(self, templates, inventory):
        """ preprocess the templates """
        cache_key = tuple(sorted(templates + [inventory]))
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]

        keys = self.template_mapper.keys() # {d} {f} {s} {o} {x}
        tp = [cmd.replace('{oven}','oven').replace('{stove}','stove').replace('{toaster}','BBQ')
              for cmd in templates]
        tp = [self.command_parser(cmd) for cmd in tp] # 转换成{'verb': 'chop', 'entity': '{f}', 'preposition': 'with', 'entity2': '{o}'}
        tp = [p for p in tp if '{' not in p['entity2'] or p['entity2'] in keys ]
        # taku: 将所有chop{f} with {o}变成chop {f} with knife
        if 'knife' in inventory:
            for template in tp:
                if template['verb'] in common.KNIFE_VERBS:
                    template['entity2'] = 'knife'
                    # 如果entity1不在inventory中，过滤掉
        else: # 如果没有刀，全部过滤
            tp = [p for p in tp if p['verb'] not in common.KNIFE_VERBS]
        # taku: 去掉所有put，因为没有必要
        tp = [p for p in tp if p['verb'] != 'put']
        # print(tp)
        out = []
        details = []
        for p in tp:
            if '{' in p['entity2']: # 确保with {o}的情况下，{o}是有值的，但是没能对应到chop {f} with {o}
                p['entity2'] = ''
                p['preposition'] = ''
            if p['entity']:
                out.append('{} {} {} {}'.format(p['verb'], p['entity'], p['preposition'], p['entity2']).strip())
                details.append(p)
            else:
                # print(p)
                pass
        self.template_cache[cache_key] = (out, details)
        return out, details

    def get_ent_types(self, cat):
        output = []
        for k, values in self.template_mapper.items():
            if cat in values:
                output.append(k)
        return sorted(output)

    def generate_all(self, entities, templates, inventory):
        """ generates candidate commands based on the the entities and
        command templates
        """
        templates, details = self.filter_templates(templates, inventory)
        # print(templates)
        # print(entities)
        output = []
        for ent, cat in entities:
            etyps = self.get_ent_types(cat)
            for tpl, detail in zip(templates, details):
                for etyp in etyps:
                    if etyp in tpl:
                        filled_template = tpl.replace(etyp, ent)
                        # 如果是用刀的动作，判断entity是否在inventory中
                        if detail['preposition'] == 'with':
                            # print(detail)
                            # print(inventory)
                            if ent not in inventory:
                                continue
                        output.append(filled_template) # dice purple potato with knife
        entity_names = [e for e,_ in entities]
        for ent in ['north', 'south', 'east', 'west']:
            if ent in entity_names:
                output.append('go {}'.format(ent))
        output.append('prepare meal')
        # print(output)
        return list(set(output))

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

def get_labels():
    #return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
    return ["O", "B-D", "I-D",  "B-C", "I-C", "B-W", "I-W", "B-S", "I-S", "B-T", "I-T", "B-F", "I-F",
            "X", "[CLS]", "[SEP]"]

class Ner:

    def __init__(self, model_dir: str = '/home/taku/Downloads/cog2019_ftwp/trained_models/bert_ner/bert_ner.tch', device: str = 'cuda'):
        self.device = device
        self.model, self.tokenizer = load_trained_model(model_dir)
        self.max_seq_length = 128
        self.model.eval()
        # self.label_map = {label : i for i, label in enumerate(get_labels(),1)}
        self.label_map = {i:label for i, label in enumerate(get_labels(),1)}

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
        input_ids = torch.tensor([input_ids],dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long).to(self.device)
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
        self.ner =  Ner()

    def entities_mapping(self, entities):
        res = []
        for e,t in entities:
            for k in self.hifen_rev_map.keys():
                if k in e:
                    e = e.replace(k, self.hifen_rev_map[k])
            res.append((e,t))
        return res

    def generate_entities(self, description, inventory):
        entities = extract_entities(description, inventory, model = self.ner) # 通过环境描述和库存信息提取实体
        return self.entities_mapping(entities)


    def preprocess_description(self, description):
        mobj = self.rgx.search(description)
        if mobj:
            kw = mobj.group(0) # 应该是匹配到的第一个带有-的词 a-b
            # dbg('Found hifen: {}'.format(kw))
            target = kw.replace('-', ' ') # a b
            self.hifen_map[kw] = target
            self.hifen_rev_map[target] = kw
            return description.replace(kw, target)
        return description

    def commands_generate(self, infos, need_taku_filter = True):
        """
        注意，这里做了一些必要的处理
        1) 将{oven}, {stove}之类的替换成具体的名字
        2) 将带有-的词分开
        3) 将cut chop slice with {o} 替换成 with knife
        4) 如果物品栏中没有a, 不需要生成verb a with b这样的指令
        5) 如果物品栏中没有knife, 不需要生成verb a with knife这样的指令
        如果 need_taku_filter = True, 调用common.filter_commands_default(commands)过滤指令
        """
        description = self.preprocess_description(infos['description'])
        inventory = infos['inventory']
        entity_types = self.generate_entities(description, inventory) # 通过ner模型，提取实体
        # entities = [e for e,_ in entity_types] 
        templates = infos['command_templates']
        commands = self.generate_commands_by_template(description, templates, entity_types, inventory) # 通过实体和菜单生成指令，为什么要菜单？
        if need_taku_filter:
            commands = common.filter_commands_default(commands)
        return commands

    def generate_commands_by_template(self, description, command_templates, entities, inventory):
        # description = self.preprocess_description(infos['description'])
        commands = self.get_admissible_commands(description, entities, command_templates, inventory) # Q: command_templates怎么来的？
        return commands


    def default_template_mapper(self):
        langmodel = CommandModel()
        return langmodel

    # NOTE: 这里没用到inventory，但是上面的genrate_entities用到了
    def get_admissible_commands(self, description, entities, templates, inventory): 
        state_entities = entities
        # dbg('State entities: {}'.format(sorted(state_entities)))
        cmds = self.langmodel.generate_all(state_entities, templates, inventory) # 通过langmodel和识别出来的实体生成所有可能的指令
        if 'cookbook' in description and 'examine cookbook' not in cmds:
            dbg('Adding examine cookbook')
            cmds.append('examine cookbook')
        return cmds
    

# 确认所有command在ner生成的指令之中
def test_walkthrough():
    import pandas as pd
    import ast
    cg = CommandGenerator()
    df = pd.read_csv('exp/auto_filename/walkthrough_valid.csv')
    counter = 0
    for index, row in tqdm(df.iterrows()):
        dict = row.to_dict()
        dict['command_templates'] = ast.literal_eval(dict['command_templates'])
        commands = cg.commands_generate(dict, need_taku_filter=True)
        cmd = row['command']
        if cmd not in commands:
            counter += 1
            dbg(f'Game name: {row["gamename"]}')
            dbg(f'Missing command: {cmd}')
            dbg(f'Commands: {commands}')
            dbg(f'Description: {row["description"]}')
            dbg(f'Inventory: {row["inventory"]}')
            dbg('------------------------------------')
    dbg(f'Total missing commands: {counter}/{len(df)}')
