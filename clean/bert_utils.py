from transformers import BertForMaskedLM, AutoTokenizer
from functools import lru_cache
import torch
from game import Game_state
import logging
from recordclass import recordclass

NextCommandResult = recordclass('NextCommandResult', 'index command logits distribution')

DEBUG = True
if DEBUG:
    logging.basicConfig(filename='log/global.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('bert_utils')
dbg = logger.debug

BERT_BASE_UNCASED_MODEL_ID = 'bert-base-uncased'
history_window=20

@lru_cache(maxsize=1)
def default_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)

def init_bert_ours():
    return BertForMaskedLM.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)

def bert_prompt_from_game_state(game_state: Game_state, need_action_history = True, history_window = 100, seperater = '[SEP]'):
    x = ''
    x += f"Room: {game_state.room} {seperater}"
    x += f"Recipe: {game_state.recipe_clean()} {seperater}"
    x += f"Inventory: {game_state.inventory_clean()} {seperater}" # NOTE: 2025.3.18 增加以平等测试
    if need_action_history:
        action_history_text = game_state.action_history(history_window=history_window)
        x += f"Action history: {action_history_text} {seperater}" 
    available_commands_text = game_state.available_commands_text()
    x += f'Available actions:\n{available_commands_text}'
    return x


@lru_cache(maxsize=None)
def command_indexs_tokenized(command_length = 100):
    tokenizer = default_tokenizer()
    command_index_string = ' '.join([str(item) for item in list(range(command_length))])
    results =  tokenizer.encode(command_index_string, add_special_tokens = False)
    assert len(results) == command_length, f"command_indexs_tokenized: {len(results)} != {command_length}"
    return results


def tokenize_game_state(game_state: Game_state):
    x = bert_prompt_from_game_state(game_state, need_action_history=True, history_window=history_window)
    tokenizer = default_tokenizer()
    prompt_ids = tokenizer('[CLS] ' + x + ' [SEP]', return_tensors="pt", add_special_tokens=False)["input_ids"] # [[token_length]]
    if prompt_ids.shape[1] > 480:
        dbg('过长prompt需要进行截断: ')
        dbg(x)
        prompt_ids = prompt_ids[:, :480]
    return prompt_ids # [1, token_length]

# @parameter: x是包含[MASK]标记的prompt
def get_mask_logits_simple(bert, state: Game_state):
    prompt_ids = tokenize_game_state(state)
    with torch.no_grad():
        logits = bert(input_ids=prompt_ids.to(bert.device)).logits # (batch_size, seq_length, vocab_size)
    mask_token_index = (prompt_ids == default_tokenizer().mask_token_id)[0].nonzero(as_tuple=True)[0] # TODO: 检查
    return logits[0, mask_token_index] # (30522)

# @parameter: x是包含[MASK]标记的prompt
def get_cls_logits_simple(bert, state: Game_state):
    prompt_ids = tokenize_game_state(state) # [[token_length]]
    with torch.no_grad():
        logits = bert(input_ids=prompt_ids.to(bert.device)).logits
    cls_token_index = 0
    return logits[0, cls_token_index] # (30522)

# 用于critic
# @parameter: x是包含[MASK]标记的prompt
def get_cls_output(model, x):
    device = model.device
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    out = model(**inputs.to(device), output_hidden_states=True) # 23 layers tuple
    last_layer = out.hidden_states[-1] # (1, 52, 768)
    cls_out = last_layer[:, 0] # (1, 768)
    return cls_out


def get_command_logits_simple(model, state: Game_state, from_cls = True):
    if not from_cls:
        mask_logits = get_mask_logits_simple(model, state) # (1, 50368)
    else:
        mask_logits = get_cls_logits_simple(model, state) # (1, 50368)
    command_length = len(state.filtered_available_commands())
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = mask_logits[command_indexs] # (command_length)
    return command_logits # (command_length)

def test():
    from game import default_game
    from model_ours import Model
    model = Model()
    model.init_bert()
    game = default_game()
    _ = game.reset()
    game.act('go east')
    state = game.to_game_state()
    return model.bert, state

def get_command_distribution_simple(model, state: Game_state):
    command_logits = get_command_logits_simple(model, state)
    # print(command_logits) # NOTE: TESTING
    command_logits[command_logits < 0] = 0 # 出现负数无法用于建构distribution，会报错，因此直接设置为0即可
    import torch
    dist = torch.distributions.categorical.Categorical(probs = command_logits)
    return dist


# 拥有探索性
def get_next_command_by_distribution(model, state: Game_state):
    dist = get_command_distribution_simple(model, state)
    command_index = dist.sample().item()
    command = state.filtered_available_commands()[command_index]
    results = NextCommandResult(command_index, command, distribution=dist)
    return results


# 贪婪
def get_next_command(model, state: Game_state):
    command_logits = get_command_logits_simple(model, state) # (command_length)
    command_index = command_logits.argmax().item()
    command = state.filtered_available_commands()[command_index]
    results = NextCommandResult(command_index, command, command_logits)
    return results