from bert_common import Abs_critic, Game_state, replace_mask, initiate_bert
import torch.nn as nn

def squared_loss(a, b):
    return -1

def encode_get_cls(bert, x):
    pass

def expect_score(mlp_scorer, cls_token):
    pass

class MLP_scorer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP_scorer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 只输出一个值
        )
    def forward(self, x):
        return self.network(x)  # 输出一个标量
    def expect_score(self, cls_token):
        return self.forward(cls_token) # 期待输入为(768)，输出为(1)

class Critic(Abs_critic):
    def __init__(self) -> None:
        self.mlp_scorer = None
        self.bert = None
        self.accelerator = None
        self.optimizer = None
    def init_model(self):
        self.bert = initiate_bert()
        self.mlp_scorer = MLP_scorer(768, 512)
        # TODO: 初始化accelerator和optimizer
    def expect_return(self, state: Game_state, action: str):
        x = state.x
        action_list = state.action_list
        # 使用BERT的cls_token来进行分数的评定
        action_index = action_list.index(action)
        # 将x中的[MASK]替换成action_index
        x = replace_mask(x, action_index) # TEST DONE
        cls_token = encode_get_cls(self.bert, x) # (1, 768)
        expect_score = self.mlp_scorer.expect_score(cls_token) # (1) 类似于 [0.0784]
        return expect_score
    def update_critic(self, loss):
        self.accelerator.backward(loss)
        self.optimizer.step()

# 实现最简单的QAC算法
def qac_one_episode(game, actor, critic):
    s0 = game.get_state()
    a0 = actor.next_action(s0)
    r = game.act(a0)
    s1 = game.get_state()
    a1 = actor.next_action(s1)
    # update critic
    # critic.expect_return(s0, a0) = critic.expect_return(s1, a1) + r
    critic_loss = squared_loss(critic.expect_return(s0, a0), critic.expect_return(s1, a1) + r)
    critic.update_critic(critic_loss)
    # update actor
    actor_loss_number = actor.action_select_loss(s0, a0, critic.expect_return(s0, a0))
    actor.update_policy()
 

def for_test():
    from bert_common import initiate_bert
    bert = initiate_bert()
    from bert_for_ftwp import game_for_test
    game = game_for_test()