from functools import lru_cache
import torch
from torch import nn
import common
from common import draw_line_chart
import numpy as np
from interface_ftwp import Ftwp_interface_by_path

class Game_state:
    def __init__(self) -> None:
        self.x = ''
        self.action_list = []

class Abs_model_policy_gradient:
    def clean_gradient(self):
        pass
    def next_action(self, state: Game_state):
        return 'look'
    def update_policy(self, state: Game_state, action, reward_scalar):
        # 需要观察之前的梯度下降是怎样和训练库联动的
        pass
    def action_select_loss(self, state: Game_state, action, reward_scalar, ):
        return 0
    
class Abs_critic:
    def expect_return(self, state: Game_state, action: str):
        return -1
    def update_critic(self, loss):
        pass

def get_optimizer(model_or_paras):
    if getattr(model_or_paras, "parameters", None):
        return torch.optim.AdamW(model_or_paras.parameters(), lr=2e-5)
    else:
        return torch.optim.AdamW(model_or_paras, lr=2e-5)

@lru_cache(maxsize=None)
def default_tokenizer():
    from transformers import AutoTokenizer
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

@lru_cache(maxsize=None)
def default_mse_loss():
    return nn.MSELoss()

def initiate_bert():
    from transformers import ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    model = ModernBertForMaskedLM.from_pretrained(model_id)
    model = model.train()
    return model


def replace_mask(x: str, action_index: int):
    return x.replace('[MASK]', str(action_index))

def get_loss(model, x: str, y : int , device = 'cpu'):
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    # labels = x.replace('[MASK]', str(y))
    labels = replace_mask(x, y)
    labels = tokenizer(labels, return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(**inputs.to(device), labels=labels.to(device))
    return outputs.loss


class Logger_loss_and_score:
    def __init__(self, file_name = ''):
        self.temp_losses = []
        self.temp_rewards = []
        self.losses_per_episode = []
        self.rewards_per_episode = []
        self.counter = 0
        self.global_counter = 0
        self.file_name = file_name or 'default'
        self.checkpoint_path = f'exp/auto_filename/{self.file_name}.tch'
        self.image_path = f'exp/auto_filename/{self.file_name}.png'
        self.text_log_path = f'exp/auto_filename/{self.file_name}.txt'
        self.text_log = ''
    def add_loss(self, loss):
        self.temp_losses.append(loss)
    def add_reward(self, reward):
        self.temp_rewards.append(reward)
    def get_losses(self):
        return self.losses
    def draw_only(self):
        draw_line_chart(list(range(len(self.losses_per_episode))), [self.losses_per_episode, self.rewards_per_episode], ['mean loss', 'reward'], path=self.image_path)
    def episode_log(self):
        # calculate episode loss and total reward
        episode_avg_loss = np.mean(self.temp_losses)
        episode_total_reward = np.sum(self.temp_rewards)
        self.losses_per_episode.append(episode_avg_loss)
        self.rewards_per_episode.append(episode_total_reward)
        # clean temp losses and rewards
        self.temp_losses = []
        self.temp_rewards = []
        self.draw_only()


# =============== 通用指令获取 =====================

@lru_cache(maxsize=None)
def command_indexs_tokenized(command_length = 100):
    tokenizer = default_tokenizer()
    command_index_string = ' '.join([str(item) for item in list(range(command_length))])
    return tokenizer.encode(command_index_string, add_special_tokens = False)

# TODO: 代换掉原来的函数
# @parameter: x是包含[MASK]标记的prompt
def get_mask_logits_simple(model, x):
    import torch
    device = model.device
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    return logits[0, mask_token_index] # (1, 50368)

# 用于critic
# @parameter: x是包含[MASK]标记的prompt
def get_cls_output(model, x):
    import torch
    device = model.device
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    out = model(**inputs.to(device), output_hidden_states=True) # 23 layers tuple
    last_layer = out.hidden_states[-1] # (1, 52, 768)
    cls_out = last_layer[:, 0] # (1, 768)
    return cls_out


# 需要从game处获得filtered_commands，那可以让他直接传入
""" def get_command_logits(game, tokenizer, model):
    mask_logits = get_mask_logits(game, tokenizer, model) # (1, 50368)
    command_length = len(game.filtered_commands)
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = mask_logits[0, command_indexs] # (command_length)
    return command_logits # (command_length)
 """
def get_command_logits_simple(model, state: Game_state):
    mask_logits = get_mask_logits_simple(model, state.x) # (1, 50368)
    command_length = len(state.action_list)
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = mask_logits[0, command_indexs] # (command_length)
    return command_logits # (command_length)

""" def get_command_distribution(game, tokenizer, model):
    command_logits = get_command_logits(game, tokenizer, model)
    command_logits[command_logits < 0] = 0 # 出现负数无法用于建构distribution，会报错，因此直接设置为0即可
    # print('=========')
    # print(command_logits) # NOTE: need test
    # print('=========')
    import torch
    dist = torch.distributions.categorical.Categorical(probs = command_logits)
    return dist """

def get_command_distribution_simple(model, state: Game_state):
    command_logits = get_command_logits_simple(model, state)
    # print(command_logits) # NOTE: TESTING
    command_logits[command_logits < 0] = 0 # 出现负数无法用于建构distribution，会报错，因此直接设置为0即可
    import torch
    dist = torch.distributions.categorical.Categorical(probs = command_logits)
    return dist

# NOTE: Added 2025.2.12 理论上可以根绝错误输出的问题
""" def get_next_command_by_distribution(game, tokenizer, model):
    dist = get_command_distribution(game, tokenizer, model)
    command_index = dist.sample().item()
    command = game.filtered_commands[command_index]
    return command """

# 拥有探索性
def get_next_command_by_distribution_simple(model, state: Game_state, need_dist = False):
    dist = get_command_distribution_simple(model, state)
    command_index = dist.sample().item()
    command = state.action_list[command_index]
    if need_dist:
        return command, dist
    else:
        return command

# Added 2025.2.12 限定版的argmax
""" def get_next_command_by_command_logits_argmax(game, tokenizer, model):
    command_logits = get_command_logits(game, tokenizer, model) # (command_length)
    command_index = command_logits.argmax().item()
    command = game.filtered_commands[command_index]
    return command """

# 贪婪
def get_next_command_by_command_logits_argmax_simple(model, state: Game_state):
    command_logits = get_command_logits_simple(model, state) # (command_length)
    command_index = command_logits.argmax().item()
    command = state.action_list[command_index]
    return command


# =============== 把Game类般进来 =================

class Prompt_builder_for_bert:
    def __init__(self):
        self.room_name = ''
        self.action_history = ''
        self.action_list = ''
        # others
        self.recipe = ''
        self.prompt = ''

    def build(self):
        user_msg = ''
        user_msg += f"Room: {self.room_name if self.room_name else 'Unknown now'}\n"
        user_msg += f"Recipe: {self.recipe if self.recipe else 'Unknown now'}\n"
        user_msg += f"Action history: {self.action_history if self.action_history else ''}\n" 
        user_msg += f'Available actions:\n{self.action_list}\n' if self.action_list else ''
        user_msg += 'Next action: [MASK]'
        self.prompt = user_msg

def prompt_from_env_feedback_action_template(description, action_obs_pairs, available_actions, recipe):
    promptor = Prompt_builder_for_bert()
    promptor.action_history = common.action_obs_pairs_to_history(action_obs_pairs, seperator='>')
    promptor.room_name = common.extract_room_name(description)
    promptor.action_list = common.actions_to_list_number(available_actions)
    promptor.recipe = recipe
    promptor.build()
    return promptor.prompt

def bert_prompt_from_game(game):
    return prompt_from_env_feedback_action_template(game.description, game.action_obs_pairs, game.available_actions, game.recipe)

class Game_for_bert(Ftwp_interface_by_path):
    def init_hook(self):
        pass
    def input(self, command):
        self.command = command
        try:
            x = bert_prompt_from_game(self)
            y = self.available_actions.index(command)
            self.finetune_triples.append((x, y)) # save x and y
            self.act_and_output(command)
        except:
            print(f'指令不存在，不需要执行: {command}')
            print(self.available_actions)
            print(self.filename)
    def save_as_json(self):
        f = open(f'exp/auto_filename/{self.filename}', 'w')
        for x, y in self.finetune_triples:
            obj = {'x': x.strip(), 'y': str(y).strip()}
            f.write(str(obj) + '\n')
        f.close()
    def output_actions(self):
        actions = [y for x, y in self.finetune_triples]
        return actions
    def save_readable(self):
        filename = self.filename + f'score{self.get_score()}_of_{self.get_max_score()}'
        f = open(f'exp/auto_filename/{filename}.txt', 'w')
        for x, y in self.finetune_triples:
            f.write(f'{x}\n\n{y}\n\n')
        f.write(f'{self.get_score()}/{self.get_max_score()}')
        f.close()
    def filter_walkthroughs(self, walkthroughs = None):
        walkthroughs = walkthroughs or self.get_walkthrough()
        filtered_walkthroughs = []
        for action in walkthroughs:
            if action in self.filtered_commands:
                filtered_walkthroughs.append(action)
            else:
                print(f'Action invalid: {action}')
        return filtered_walkthroughs
    def filter_walkthroughs_with_input_test(self, walkthroughs = None):
        walkthroughs = walkthroughs or self.get_walkthrough()
        self.verbose = False;
        self.reset()
        filtered_walkthroughs = []
        for action in walkthroughs:
            if action in self.filtered_commands:
                filtered_walkthroughs.append(action)
                self.input(action)
            else:
                print(f'filtered action: {action}')
        self.reset()
        return filtered_walkthroughs
    def auto_play(self, action_list):
        self.reset()
        for action in action_list:
            self.input(action)
    def get_x(self):
        return bert_prompt_from_game(self)
    def action_obs_pairs_got_callback(self, action_obs_pairs):
        self.save_if_checking_recipe(action_obs_pairs) 
        self.save_instant_reward(action_obs_pairs)
    def save_instant_reward(self, action_obs_pairs):
        self.instant_reward = 0
        if not action_obs_pairs or len(action_obs_pairs) < 1:
            return
        act, obs = action_obs_pairs[-1]
        if 'Your score has just gone up by one point' in obs:
            self.instant_reward = 1
    def get_instant_reward(self):
        return self.instant_reward
    

class Game_for_rl(Game_for_bert):
    def get_state(self):
        state = construct_game_state(self)
        return state
    def act(self, action):
        if action not in self.filtered_commands:
            print('???????????????????')
            print(action)
            print(self.filtered_commands)
            return 0
        self.input(action)
        reward = self.get_instant_reward()
        return reward
    


# ================ 通用函数 ==================

def load_trained_model(path):
    from transformers import AutoTokenizer, ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBertForMaskedLM.from_pretrained(path)
    _ = model.cuda()
    return model, tokenizer

def construct_game_state(game: Game_for_bert):
    game_state = Game_state()
    game_state.x = game.get_x()
    game_state.action_list = game.filtered_commands
    return game_state

# ================ For Test & Valid ===================

# ==== auto play ========


def trained_model_autoplay(game, bert, save_readable = True, sample_command_func = get_next_command_by_command_logits_argmax_simple):
    bert.eval()
    game.reset()
    counter = 0
    while not any([counter >= 50, game.is_won(), game.is_lost()]):
        # command = get_next_command(game, tokenizer, model)
        # command = get_next_command_by_distribution(game, tokenizer, model) # NOTE: 2025.2.12 用sample取代原来的argmax
        # command = get_next_command_by_command_logits_argmax(game, tokenizer, model) # NOTE: 2025.2.12 用sample取代原来的argmax
        game_state = construct_game_state(game)
        command = sample_command_func(bert, game_state)
        game.input(command)
        counter += 1
    if save_readable:
        game.save_readable()
    if counter >= 30:
        print('测试时步数大于30!')
    return game.get_score(), game.get_max_score()

# 首先要获得x, 然后通过model获得logits输出，然后获得mask对应的logits
""" def get_mask_logits(game, tokenizer, model):
    import torch
    device = model.device
    x = game.get_x()
    inputs = tokenizer(x, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    return logits[0, mask_token_index] # (1, 50368) """

# @history: 1.17: 需要分开彻底分开
def valid_paths():
    from ftwp_info import all_valid_game_paths
    return all_valid_game_paths(shuffle = True)[-30:]

def batch_test(model, save_readable = True, test_game_paths = [], file_prefix = '', need_kitchen_visit_rate = False, txtLogger = common.Fake_text_logger()):
    if len(test_game_paths) < 1:
        test_game_paths = valid_paths()
    scores = []
    max_scores = []
    # 排除没有访问到kitchen的scores和max_scores并计算该情况的得分
    scores_without_kitchen = []
    max_scores_without_kitchen = []
    kitchen_visited_counter = 0
    for path in test_game_paths:
        game = Game_for_bert(path)
        game.verbose = False
        game.filename = file_prefix + game.filename # Added 2025.2.8
        score, max_score = trained_model_autoplay(game, model, save_readable)
        if not game.kitchen_visited:
            txtLogger.add(f'Score at {score} & kitchen not visited: {game.filename}')
            txtLogger.write_txt_log()
        else:
            scores_without_kitchen.append(score)
            max_scores_without_kitchen.append(max_score)
        kitchen_visited_counter += game.kitchen_visited
        scores.append(score)
        max_scores.append(max_score)
    score_final = sum(scores) / sum(max_scores)
    if not need_kitchen_visit_rate:
        return score_final
    else:
        kitchen_visited_rate = kitchen_visited_counter / len(test_game_paths)
        scores_without_kitchen = sum(scores_without_kitchen) / sum(max_scores_without_kitchen)
        txtLogger.add(f'score: {score_final}')
        txtLogger.add(f'kitchen_rate: {kitchen_visited_rate}')
        txtLogger.add(f'score_with_kitchen: {scores_without_kitchen}')
        txtLogger.write_txt_log()
        return score_final, scores_without_kitchen, kitchen_visited_rate

def batch_valid(model, save_readable = True):
    return batch_test(model, save_readable = save_readable, test_game_paths=valid_paths())


def run_test_full(model, file_prefix = ''):
    from ftwp_info import all_test_game_paths
    txtLogger = common.Logger_simple(file_name=f'run_test_full_{file_prefix}_log')
    test_game_paths=  all_test_game_paths()
    return batch_test(model, save_readable=False, test_game_paths=test_game_paths, need_kitchen_visit_rate = True, txtLogger=txtLogger)

def run_valid_full(model, file_prefix = ''):
    from ftwp_info import all_valid_game_paths
    txtLogger = common.Logger_simple(file_name=f'run_valid_full_{file_prefix}_log')
    valid_game_paths=  all_valid_game_paths()
    return batch_test(model, save_readable=False, test_game_paths=valid_game_paths, need_kitchen_visit_rate = True, txtLogger=txtLogger)

# 2025.3.3 
def game_for_train(game_index = 0):
    from ftwp_info import all_train_game_paths
    file_paths = all_train_game_paths()
    game = Game_for_rl(file_paths[game_index])
    game.verbose = True
    return game

# ================== For Command probabilty analysis ================

def game_play_analysis(bert, game, txtLogger = common.Logger_simple()):
    bert.eval()
    game.reset()
    counter = 0
    while not any([counter >= 50, game.is_won(), game.is_lost()]):
        # command = get_next_command(game, tokenizer, model)
        # command = get_next_command_by_distribution(game, tokenizer, model) # NOTE: 2025.2.12 用sample取代原来的argmax
        # command = get_next_command_by_command_logits_argmax(game, tokenizer, model) # NOTE: 2025.2.12 用sample取代原来的argmax
        game_state = construct_game_state(game)
        txtLogger.add(game_state.x)
        dist = get_command_distribution_simple(bert, game_state)
        command_probs = dist.probs
        common.beutiful_print_command_and_probs(game_state.action_list, command_probs, txtLogger.add)
        command = game_state.action_list[dist.probs.argmax().item()]
        game.input(command)
        counter += 1
    txtLogger.write_txt_log()
    return game.get_score(), game.get_max_score()


# ============== Lora ================

def initiate_lora_bert():
    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0, target_modules=["Wqkv"])
    bert = initiate_bert()
    lora_bert = get_peft_model(bert, peft_config)
    print('初始化lora模型！')
    lora_bert.print_trainable_parameters()
    return lora_bert

def save_lora_bert(lora_bert, output_dir = 'exp/auto_filename/dd.tch'):
    lora_bert.save_pretrained(output_dir)

def mark_lora_require_grad(model):
    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True

def load_lora_bert(output_dir = 'exp/auto_filename/dd.tch', training = False):
    from peft import AutoPeftModel
    model = AutoPeftModel.from_pretrained(output_dir)
    print('加载训练好的lora模型！')
    if training:
        mark_lora_require_grad(model)
    model.print_trainable_parameters()
    return model