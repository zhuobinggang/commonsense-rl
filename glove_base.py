import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
import torchtext
from functools import lru_cache
from typing import List

# ============ 公用函数 ============
@lru_cache(maxsize=None)
def default_tokenizer():
    tokenizer = get_tokenizer("basic_english")
    return tokenizer

@lru_cache(maxsize=None)
def glove_vocab(dim = 100):
    vec = torchtext.vocab.GloVe(name='6B', dim=dim, cache = '.vector_cache')
    return vec

# return size: (tokens, dim)
def text_to_tensor(text, dim = 100):
    tokens = default_tokenizer()(text)
    vec = glove_vocab(dim = dim)
    return vec.get_vecs_by_tokens(tokens, lower_case_backup=True)


class GRUPolicy(nn.Module):
    def __init__(self, embed_dim = 100, hidden_dim = 64):
        super(GRUPolicy, self).__init__()
        # GloVe Embedding Layer (Frozen)
        # glove_weights = torch.load(f".vector_cache/glove.6B.{embed_dim}d.txt.pt")
        # self.embedding = nn.Embedding.from_pretrained(glove_weights[2]) # (400k, 100)
        # self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # GRU Encoders
        self.env_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.cmd_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # Fully connected layer for action selection
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Concatenated features
        self.init_optimizer()
        
    def init_optimizer(self): # TODO: 使用accelarate库
            self.optimizer = optim.Adam(self.parameters(), lr=5e-4)

    def encode_text(self, text_tensor, gru):
        # embeds = self.embedding(text_tensor)
        text_tensor = text_tensor.unsqueeze(0) # (tokens, dim) -> (1, tokens, dim)
        _, hidden = gru(text_tensor) # (1, tokens, dim) -> (1, hidden_dim) 
        return hidden.squeeze(0)  # (1, hidden_dim) -> (hidden_dim)
    
    def forward(self, env_desc: str, cmd_list: List[str]) -> torch.Tensor:
        env_desc = text_to_tensor(env_desc) # (tokens, dim)
        cmd_list = [text_to_tensor(cmd) for cmd in cmd_list] # (tokens, dim)
        env_encoded = self.encode_text(env_desc, self.env_gru)  # (hidden_dim)
        cmd_encoded = torch.stack([self.encode_text(cmd, self.cmd_gru) for cmd in cmd_list], dim=1)  # (num_cmds, hidden_dim)
        
        # Concatenate environment and command encodings
        env_expanded = env_encoded.expand_as(cmd_encoded)  # (num_cmds, hidden_dim) 复制
        concat_features = torch.cat([env_expanded, cmd_encoded], dim=1)  # (num_cmds, hidden_dim * 2)
        
        # Compute logits and probabilities
        logits = self.fc(concat_features)  # (num_cmds, 1)
        logits = logits.squeeze(1)  # (num_cmds)
        probs = F.softmax(logits)  # Convert to probabilities, (num_cmds)
        
        return probs
    
    def clean_gradient(self): # TODO
        pass
    def next_action(self, state):
        env_desc = state.x_without_commands()
        cmd_list = state.get_commands()
        with torch.no_grad():
            probs = self(env_desc, cmd_list)
        max_prob_index = torch.argmax(probs).item()
        return cmd_list[max_prob_index]
    def update_policy(self, state, action, reward_scalar): # TODO
        # 需要观察之前的梯度下降是怎样和训练库联动的
        pass
    def action_select_loss(self, state, action, reward_scalar):
        env_desc = state.x_without_commands()
        cmd_list = state.get_commands()
        selected_cmd_idx = cmd_list.index(action)
        probs = self.forward(env_desc, cmd_list) # (num_cmds)
        log_prob = torch.log(probs[selected_cmd_idx])
        loss = -1 * reward_scalar * log_prob
        return loss

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
