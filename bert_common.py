from functools import lru_cache
import torch
from common import draw_line_chart
import numpy as np

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
    def stack_loss(self, state: Game_state, action, reward_scalar):
        return 0

def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), 2e-5)

@lru_cache(maxsize=None)
def default_tokenizer():
    from transformers import AutoTokenizer
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def initiate_bert():
    from transformers import ModernBertForMaskedLM
    model_id = "answerdotai/ModernBERT-base"
    model = ModernBertForMaskedLM.from_pretrained(model_id)
    model = model.train()
    return model

def get_loss(model, x: str, y : int , device = 'cpu'):
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    labels = x.replace('[MASK]', str(y))
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
