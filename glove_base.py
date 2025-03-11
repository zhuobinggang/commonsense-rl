import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

# from torchtext.vocab import GloVe
from torchtext_utils import get_tokenizer
from functools import lru_cache
from typing import List
import common

from bert_common import first_train_game
import bert_common

# ============ 公用函数 ============
@lru_cache(maxsize=None)
def default_tokenizer():
    tokenizer = get_tokenizer("basic_english")
    return tokenizer

def preprocess_and_save_vocab_embeddings(dim = 100):
    vocab,embeddings = [],[]
    with open(f'.vector_cache/glove.6B.{dim}d.txt','rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
    # turn into numpy
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    # print(vocab_npa[:10])
    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    # load into torch
    # my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
    # return my_embedding_layer, vocab_npa
    with open(f'.vector_cache/glove.6B.{dim}d.vocab.npy','wb') as f:
        np.save(f,vocab_npa)
    with open(f'.vector_cache/glove.6B.{dim}d.embeddings.npy','wb') as f:
        np.save(f,embs_npa)
    return None

@lru_cache(maxsize=None)
def load_vocab(dim = 100):
    with open(f'.vector_cache/glove.6B.{dim}d.vocab.npy','rb') as f:
        vocab = np.load(f)
    return vocab

@lru_cache(maxsize=None)
def load_embeddings(dim = 100, into_torch = False):
    with open(f'.vector_cache/glove.6B.{dim}d.embeddings.npy','rb') as f:
        embeddings = np.load(f)
    if into_torch:
        return torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float()).cuda()
    return embeddings


@lru_cache(maxsize=None)
def glove_vocab(dim = 100):
    vocab = load_vocab(dim)
    return vocab

# return size: (tokens, dim)
def text_to_idx(text, dim = 100):
    tokens = default_tokenizer()(text)
    vocab = glove_vocab(dim = dim)
    # get index for each token, if not found, use <unk> token
    idxs = []
    for token in tokens:
        try:
            idx = vocab.index(token)
        except:
            idx = 1
        idxs.append(idx)
    return torch.tensor(idxs).cuda()


def text_to_tensor(text, dim = 100):
    idxs = text_to_idx(text, dim)
    embs = load_embeddings(dim, into_torch = True)
    return embs(idxs) # (tokens, dim)

# ============ prompt builder ============


# ============ Policy ============
class GRUBaseline(nn.Module):
    def __init__(self, embed_dim = 100, hidden_dim = 64, hidden_linear_size = 128):
        super(GRUBaseline, self).__init__()
        # GRU Encoders
        self.env_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_linear_size),  # 隐藏层 1
            nn.ReLU(),
            nn.Linear(hidden_linear_size, 1)
        )
        self.cuda()
    
    def init_optimizer(self): # TESTING: 使用accelarate库
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        self, optimizer = accelerator.prepare(
            self, optimizer
        )
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.prepared_for_training = True

    def encode_text(self, text_tensor, gru):
        # embeds = self.embedding(text_tensor)
        text_tensor = text_tensor.unsqueeze(0) # (tokens, dim) -> (1, tokens, dim)
        _, hidden = gru(text_tensor) # (1, tokens, dim) -> (2, hidden_dim) 这里是2因为双方向
        # assert hidden.shape == (2, 1, gru.hidden_size) 
        return hidden.reshape(-1)  # (hidden_dim * 2)
    
    def forward(self, state):
        env_desc = state.x
        return self.forward_text(env_desc)

    def forward_text(self, env_desc: str) -> torch.Tensor:
        env_desc = text_to_tensor(env_desc) # (tokens, dim)
        env_encoded = self.encode_text(env_desc, self.env_gru)  # (hidden_dim)
        return self.fc(env_encoded)  # (1)
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class GRUPolicy(nn.Module):
    def __init__(self, embed_dim = 100, hidden_dim = 64, hidden_linear_size = 128):
        super(GRUPolicy, self).__init__()
        # GRU Encoders
        self.env_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.cmd_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Fully connected layer for action selection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_linear_size),  # 隐藏层 1
            nn.ReLU(),
            nn.Linear(hidden_linear_size, 1)
        )
        self.cuda()
        self.prepared_for_training = False
        self.prefix = 'GRUPolicy'
        # self.init_optimizer()
        
    def init_optimizer(self): # TESTING: 使用accelarate库
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        optimizer = optim.AdamW(self.parameters(), lr=5e-4)
        self, optimizer = accelerator.prepare(
            self, optimizer
        )
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.prepared_for_training = True
        

    def encode_text(self, text_tensor, gru):
        # embeds = self.embedding(text_tensor)
        text_tensor = text_tensor.unsqueeze(0) # (tokens, dim) -> (1, tokens, dim)
        _, hidden = gru(text_tensor) # (1, tokens, dim) -> (2, hidden_dim) 这里是2因为双方向
        # assert hidden.shape == (2, 1, gru.hidden_size) 
        return hidden.reshape(-1)  # (hidden_dim * 2)
    
    # checked
    def forward(self, env_desc: str, cmd_list: List[str]) -> torch.Tensor:
        env_desc = text_to_tensor(env_desc) # (tokens, dim)
        cmd_list = [text_to_tensor(cmd) for cmd in cmd_list] # (cmds, tokens, dim)
        env_encoded = self.encode_text(env_desc, self.env_gru)  # (hidden_dim * 2)
        cmd_encoded = torch.stack([self.encode_text(cmd, self.cmd_gru) for cmd in cmd_list])  # (num_cmds, hidden_dim * 2)
        # Concatenate environment and command encodings
        env_expanded = env_encoded.expand_as(cmd_encoded)  # (num_cmds, hidden_dim * 2) 复制
        concat_features = torch.cat([env_expanded, cmd_encoded], dim=1)  # (num_cmds, hidden_dim * 4)
        # print(env_expanded.shape, cmd_encoded.shape, concat_features.shape, env_encoded.shape)
        # Compute logits and probabilities
        logits = self.fc(concat_features)  # (num_cmds, 1)
        logits = logits.squeeze(1)  # (num_cmds)
        probs = F.softmax(logits)  # Convert to probabilities, (num_cmds)
        assert probs.shape == (len(cmd_list),)
        return probs
    
    def clean_gradient(self): # TESTING
        if not self.prepared_for_training:
            raise Exception("Model not prepared for training")
        self.optimizer.zero_grad()
    def next_action(self, state): # only for testing
        env_desc = state.x # TESTING
        cmd_list = state.action_list
        with torch.no_grad():
            probs = self(env_desc, cmd_list)
        max_prob_index = torch.argmax(probs).item()
        return cmd_list[max_prob_index]
    def next_action_with_explore(self, state):
        env_desc = state.x # TESTING
        cmd_list = state.action_list
        with torch.no_grad():
            probs = self(env_desc, cmd_list)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        return cmd_list[action_idx.item()]
    def action_select_loss(self, state, action, reward_scalar):
        env_desc = state.x
        cmd_list = state.action_list
        selected_cmd_idx = cmd_list.index(action)
        probs = self.forward(env_desc, cmd_list) # (num_cmds)
        log_prob = torch.log(probs[selected_cmd_idx])
        loss = -1 * reward_scalar * log_prob
        return loss
    def backward_loss(self, loss):
        if not self.prepared_for_training:
            raise Exception("Model not prepared for training")
        self.accelerator.backward(loss)
    def optimizer_step(self):
        if not self.prepared_for_training:
            raise Exception("Model not prepared for training")
        self.optimizer.step()
    def save_checkpoint(self, checkpoint_path = None):
        if not checkpoint_path:
            checkpoint_path = f'exp/auto_filename/{self.prefix}.tch'
        torch.save(self.state_dict(), checkpoint_path)
    def load_checkpoint(self, checkpoint_path = None):
        if not checkpoint_path:
            checkpoint_path = f'exp/auto_filename/{self.prefix}.tch'
        self.load_state_dict(torch.load(checkpoint_path, weights_only= True))


def test_one_game(model, game):
    from bert_common import trained_model_autoplay
    return trained_model_autoplay(game, model, steps_limit=30)


def reinforce_update(policy, optimizer, env_desc, cmd_list, selected_cmd_idx, reward, baseline):
    optimizer.zero_grad()
    probs = policy(env_desc, cmd_list) # (num_cmds)
    log_prob = torch.log(probs[selected_cmd_idx])
    loss = -1 * (reward - baseline) * log_prob
    loss.backward()
    optimizer.step()
    return loss.item()
    

def example():
    # Example usage
    vocab_size = 10000
    embed_dim = 300
    hidden_dim = 128
    glove_embeddings = np.random.rand(vocab_size, embed_dim)  # Replace with real embeddings
    policy = GRUPolicy(vocab_size, embed_dim, hidden_dim, glove_embeddings)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)


class Game_for_glove(bert_common.Game_for_rl):
    def get_x(self):
        return bert_common.bert_prompt_from_game(self, with_final_hint=False)
        # return common.action_obs_pairs_to_history(self.action_obs_pairs, seperator='>')

class Tester:
    def __init__(self):
        self.policy = GRUPolicy()
        self.prefix = 'Glove_policy_test'
        # self.policy.init_optimizer()
        self.init_params_for_log()
        self.init_txt_logger()
    def init_params_for_log(self):
        # for valid
        self.counter = 0;
        self.last_valid_score = 0;
        self.max_valid_score = 0;
        self.valid_scores = [];
        self.episode_rewards = [];
        self.actor_losses = [];
        self.critic_losses = []
        self.log_steps = 100;
    def save_checkpoint(self):
        checkpoint_path = f'exp/auto_filename/{self.prefix}.tch'
        self.policy.save_checkpoint(checkpoint_path)
    def load_checkpoint(self):
        checkpoint_path = f'exp/auto_filename/{self.prefix}.tch'
        self.policy.load_checkpoint(checkpoint_path)
    def maybe_valid_and_save(self):
        from bert_common import batch_valid
        self.counter += 1
        need_valid_and_update = self.counter % self.log_steps == 0
        if need_valid_and_update: # valid然后更新max_valid_score和last_valid_score，如果分数更大，保存checkpoint
            self.txtLogger.add(f'Valid中……')
            self.last_valid_score = batch_valid(self.policy, save_readable = False)
            self.txtLogger.add(f'Valid分数为{self.last_valid_score}')
            self.txtLogger.write_txt_log()
            if self.last_valid_score > self.max_valid_score:
                self.max_valid_score = self.last_valid_score
                self.save_checkpoint()
        self.valid_scores.append(self.last_valid_score)
        if need_valid_and_update: # 绘制中间图像
            pass
    def init_txt_logger(self):
        self.txtLogger = common.Logger_simple(self.prefix)
    def draw_line_chart_no_valid(self):
        common.draw_line_chart(list(range(len(self.episode_rewards))), [self.actor_losses, self.episode_rewards], ['a_loss', 'episode_scores'], path = f'exp/auto_filename/{self.prefix}.png')
    def draw_line_chart(self):
        common.draw_line_chart(list(range(len(self.episode_rewards))), [self.actor_losses, self.valid_scores], ['a_loss', 'valid_score'], path = f'exp/auto_filename/{self.prefix}.png')
    def train_20250305(self):
        from policy_algorithms import train_one_episode
        self.policy.init_optimizer()
        from ftwp_info import all_train_game_paths
        paths = all_train_game_paths()
        game_count = len(paths)
        self.txtLogger.add('一个epoch训练开始！')
        for game_index, path in enumerate(paths):
            log_txt = f'{game_index}/{game_count}'
            print(log_txt)
            self.txtLogger.add(log_txt)
            game = Game_for_glove(path)
            # self.txtLogger.add(f'训练中: {game_index}/{game_count}')
            # 监督学习，不需要任何记录
            norm_score, mean_actor_loss = train_one_episode(self.policy, game, walkthrough=game.filter_walkthroughs_with_input_test()) 
            self.episode_rewards.append(norm_score)
            self.actor_losses.append(min(mean_actor_loss * 0.1, 1.5)) # NOTE: scale for image ouput
            self.maybe_valid_and_save() # 同时保存检查点
            if self.counter % self.log_steps == 0:
                self.draw_line_chart()
        self.draw_line_chart()
        self.txtLogger.add('一个epoch训练结束。')
        self.txtLogger.write_txt_log()
    def train_one_game(self, steps = 2000, use_walkthrough = False):
        from policy_algorithms import train_one_episode
        self.policy.init_optimizer()
        self.log_steps = 10
        for i in range(steps):
            game = first_train_game()
            game.verbose = False
            print(f'{i}/{steps}')
            if use_walkthrough:
                _, mean_actor_loss = train_one_episode(self.policy, game, walkthrough=game.filter_walkthroughs_with_input_test(), txtLogger = self.txtLogger)
                dic = test_one_game(self.policy, game)
                norm_score = dic['score'] / dic['max_score']
            else:
                norm_score, mean_actor_loss = train_one_episode(self.policy, game, txtLogger = self.txtLogger)
            print(norm_score)
            self.episode_rewards.append(norm_score)
            self.actor_losses.append(min(mean_actor_loss * 0.1, 1.5)) # NOTE: scale for image ouput
            if i % self.log_steps == 0:
                self.draw_line_chart_no_valid()
        self.draw_line_chart_no_valid()
        self.txtLogger.add('训练结束。')
        self.txtLogger.write_txt_log()


class Tester_baseline(Tester):
    def __init__(self):
        super(Tester_baseline, self).__init__()
        self.baseline = GRUBaseline()
        self.prefix = 'Glove_policy_baseline'
    def init_params_for_log(self):
        super().init_params_for_log()
        self.critic_losses = []
    def draw_line_chart_no_valid(self):
        common.draw_line_chart(list(range(len(self.episode_rewards))), [self.actor_losses, self.critic_losses, self.episode_rewards], ['a_loss', 'c_loss', 'episode_scores'], path = f'exp/auto_filename/{self.prefix}.png')
    def train_one_game(self):
        from bert_pg_baseline import train_one_episode
        self.policy.init_optimizer()
        self.baseline.init_optimizer()
        for i in range(20000):
            game = first_train_game()
            print(f'{i}/20000')
            norm_score, mean_actor_loss, mean_critic_loss, actor_idle_rate = train_one_episode(self.policy, self.baseline, game, txtLogger = self.txtLogger)
            print(norm_score)
            self.episode_rewards.append(norm_score)
            self.actor_losses.append(min(mean_actor_loss * 0.1, 1.5)) # NOTE: scale for image ouput
            self.critic_losses.append(min(mean_critic_loss * 0.5, 1.5)) # NOTE: scale for image ouput
            if i % self.log_steps == 0:
                self.draw_line_chart_no_valid()
        self.draw_line_chart_no_valid()
        self.txtLogger.add('训练结束。')
        self.txtLogger.write_txt_log()


def night_run():
    t0 = Tester_baseline()
    t0.train_one_game()
    t1 = Tester()
    t1.train_one_game()
    import os
    os.system('shutdown')
    
def test_game_reset():
    from ftwp_info import all_train_game_paths
    paths = all_train_game_paths()
    game = Game_for_glove(paths[0])
    game.verbose = True
    game.reset()
    game.input('examine cookbook')
    game.reset()
    return game
