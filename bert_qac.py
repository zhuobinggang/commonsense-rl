from bert_common import Abs_critic, Game_state, replace_mask, initiate_bert, get_optimizer, default_mse_loss
from bert_for_ftwp import get_cls_output, game_for_test
import torch.nn as nn
from common import draw_line_chart
import torch

def squared_loss(a, b):
    print(f'{a.item()} {b.item()}')
    mse_func = default_mse_loss()
    return mse_func(a, b)

def encode_get_cls(bert, x):
    return get_cls_output(bert, x)

class MLP_scorer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP_scorer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 只输出一个值
            nn.ReLU() # 增加在最后一层确保期待值大于0
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
        bert = initiate_bert()
        bert.cuda()
        mlp_scorer = MLP_scorer(768, 512)
        mlp_scorer.cuda()
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        params = list(bert.parameters()) + list(mlp_scorer.parameters())
        optimizer = get_optimizer(params)
        bert, mlp_scorer, optimizer = accelerator.prepare(
            bert, mlp_scorer, optimizer
        )
        self.accelerator = accelerator
        self.bert = bert
        self.mlp_scorer = mlp_scorer
        self.optimizer = optimizer
        self.device = device
    def expect_return(self, state: Game_state, action: str):
        x = state.x
        action_list = state.action_list
        # 使用BERT的cls_token来进行分数的评定
        action_index = action_list.index(action)
        # 将x中的[MASK]替换成action_index
        x = replace_mask(x, action_index) # TEST DONE
        cls_token = encode_get_cls(self.bert, x) # (1, 768)
        expect_score = self.mlp_scorer.expect_score(cls_token) # (1, 1) 类似于 [0.0784]
        return expect_score
    def update_critic(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

# 实现最简单的QAC算法
def qac_step(game, actor, critic, a0=None, a1=None, gamma=0.99):
    s0 = game.get_state()
    
    if a0 is None:
        a0 = actor.next_action(s0)
    
    r = game.act(a0)
    s1 = game.get_state()
    
    if a1 is None:
        a1 = actor.next_action(s1)
    
    # Update critic
    with torch.no_grad():
        target = critic.expect_return(s1, a1).detach() * gamma + r  # 避免梯度传播
    critic_loss = squared_loss(critic.expect_return(s0, a0), target)
    critic.update_critic(critic_loss)

    # Update actor
    with torch.no_grad():
        q = critic.expect_return(s0, a0).item()
    actor_loss = actor.action_select_loss(s0, a0, q)
    actor.update_policy(actor_loss)

    # Logging
    return r, actor_loss.item(), critic_loss.item()
 
def qac_one_episode(game, actor, critic, training = False):
    game.reset()
    steps = 0;
    critic_loss = 0;
    actor_loss = 0;
    episode_return = 0;
    walkthrough = game.filter_walkthroughs()
    while not any([steps >= 30, game.is_won(), game.is_lost()]):
        if training:
            if steps > len(walkthrough) - 2:
                print('最后一步不用训练')
            else:
                r, a_loss, c_loss = qac_step(game, actor, critic, walkthrough[steps], walkthrough[steps + 1])
        else:
            r, a_loss, c_loss = qac_step(game, actor, critic)
        steps += 1
        episode_return += r
        actor_loss += a_loss
        critic_loss += c_loss
    return episode_return, actor_loss / steps, critic_loss / steps

class Tester:
    def __init__(self) -> None:
        critic = Critic()
        critic.init_model()
        game = game_for_test()
        self.critic = critic
        self.game = game
        # actor
        from bert_for_ftwp import Model_policy_gradient
        actor = Model_policy_gradient(initiate_bert())
        self.actor = actor
    def test(self):
        self.episode_rewards = [];
        self.actor_losses = [];
        self.critic_losses = [];
        self.game.verbose = False;
        for e in range(40):
            # _, _ ,_ =qac_one_episode(self.game, self.actor, self.critic, training=True)
            r,a,c =qac_one_episode(self.game, self.actor, self.critic)
            self.episode_rewards.append(r)
            self.actor_losses.append(a)
            self.critic_losses.append(c)
        draw_line_chart(list(range(200)), [self.episode_rewards, self.actor_losses, self.critic_losses], ['r', 'a', 'c'])
    