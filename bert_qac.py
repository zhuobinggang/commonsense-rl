from bert_common import Abs_critic, Game_state, replace_mask, initiate_bert, get_optimizer, default_mse_loss, batch_valid, load_trained_model
from bert_common import get_cls_output
import torch.nn as nn
from common import draw_line_chart
import torch
from bert_for_ftwp import Model_policy_gradient

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
            # nn.ReLU() # 增加在最后一层确保期待值大于0
        )
    def forward(self, x):
        return self.network(x)  # 输出一个标量
    def expect_score(self, cls_token):
        return self.forward(cls_token) # 期待输入为(768)，输出为(1)
    def save(self, path = 'exp/auto_filename/default_mlp_scorer.tch'):
        torch.save(self.network.state_dict(), path)
    def load(self, path = 'exp/auto_filename/default_mlp_scorer.tch'):
        self.network.load_state_dict(torch.load(path, weights_only=True))

class Critic(Abs_critic):
    def __init__(self, bert = None, mlp_scorer = None) -> None:
        self.mlp_scorer = mlp_scorer
        self.bert = bert
        self.accelerator = None
        self.optimizer = None
        self.init_model()
    def init_model(self):
        if not self.bert:
            self.bert = initiate_bert()
        self.bert.cuda()
        if not self.mlp_scorer:
            self.mlp_scorer = MLP_scorer(768, 512)
        self.mlp_scorer.cuda()
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        params = list(self.bert.parameters()) + list(self.mlp_scorer.parameters())
        optimizer = get_optimizer(params)
        self.bert, self.mlp_scorer, optimizer = accelerator.prepare(
            self.bert, self.mlp_scorer, optimizer
        )
        self.accelerator = accelerator
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
    def eval(self):
        self.bert.eval()
    def train(self):
        self.bert.train()

# 实现最简单的QAC算法
def qac_step(game, actor, critic, a0=None, a1=None, gamma=0.95):
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
        # 打印出来看看
        print(f'Critic对{a0}的评分是{q}, 该行动的实际回报为{r}')
    actor_loss = actor.action_select_loss(s0, a0, q)
    actor.update_policy(actor_loss)

    # Logging
    return r, actor_loss.item(), critic_loss.item()


# 实现最简单的QAC算法
def qac_simple(game, actor, critic, gamma=0.95):
    s0 = game.get_state()
    a0 = actor.next_action(s0)
    r = game.act(a0)
    s1 = game.get_state()
    a1 = actor.next_action(s1)
    # Update critic
    with torch.no_grad():
        target = critic.expect_return(s1, a1).detach() * gamma + r  # 避免梯度传播
    critic_loss = squared_loss(critic.expect_return(s0, a0), target)
    critic.update_critic(critic_loss) # 问题： 每一步更新critic参数，是否会导致无法DQN中描述的难以收敛的问题？DQN使用两个神经网络来异步更新critic。
    # Update actor
    with torch.no_grad():
        q = critic.expect_return(s0, a0).item()
        # 打印出来看看
        print(f'Critic对{a0}的评分是{q}, 该行动的实际回报为{r}')
    actor_loss = actor.action_select_loss(s0, a0, q)
    actor.update_policy(actor_loss)
    # Logging
    return r, actor_loss.item(), critic_loss.item()
 
def qac_one_episode(game, actor, critic, training = False):
    steps = 0;
    critic_loss = 0;
    actor_loss = 0;
    episode_return = 0;
    walkthrough = game.filter_walkthroughs_with_input_test()
    game.reset()
    actor.train()
    critic.train()
    while not any([steps >= 50, game.is_won(), game.is_lost()]):
        if training:
            if steps > len(walkthrough) - 2:
                # print('最后一步不用训练')
                break
            else:
                r, a_loss, c_loss = qac_step(game, actor, critic, walkthrough[steps], walkthrough[steps + 1])
        else:
            r, a_loss, c_loss = qac_step(game, actor, critic)
        steps += 1
        episode_return += r
        actor_loss += a_loss
        critic_loss += c_loss
    return episode_return, actor_loss / steps, critic_loss / steps

class QAC_container:
    def __init__(self) -> None:
        self.critic = None
        self.actor = None
        print('需要使用init_QA来初始化模型，或者使用load_checkpoint来加载已经训练好的模型！')
        self.init_hook()
    def init_hook(self):
        pass
    def init_QA(self):
        # critic
        self.critic = self.init_critic()
        # actor
        self.actor = Model_policy_gradient()
    def save_checkpoint(self):
        self.actor.bert.save_pretrained(f'exp/auto_filename/QAC_Actor.tch')
        self.critic.bert.save_pretrained(f'exp/auto_filename/QAC_Critic.tch')
        self.critic.mlp_scorer.save(f'exp/auto_filename/QAC_Critic_Header.tch')
    def init_critic(self, bert = None, header = None):
        return Critic(bert, header)
    def load_checkpoint(self, path, training = True):
        actor_bert, _ = load_trained_model(f'{path}/QAC_Actor.tch')
        self.actor = Model_policy_gradient(actor_bert)
        # critic
        critic_bert, _ = load_trained_model(f'{path}/QAC_Critic.tch')
        critic_header = MLP_scorer(768, 512)
        critic_header.load(f'{path}/QAC_Critic_Header.tch')
        self.critic = self.init_critic(critic_bert, critic_header)
        # 初始化
        self.actor.bert.cuda()
        self.critic.bert.cuda()
        self.critic.mlp_scorer.cuda()
        if training:
            self.actor.bert.train()
            self.critic.bert.train()
            self.critic.mlp_scorer.requires_grad_(True)
        print('Actor和critic都加载完毕')

class Tester(QAC_container):
    def init_hook(self):
        self.init_params_for_log()
    def init_params_for_log(self):
        # for valid
        self.counter = 0;
        self.last_valid_score = 0;
        self.max_valid_score = 0;
        self.valid_scores = [];
        self.episode_rewards = [];
        self.actor_losses = [];
        self.critic_losses = [];
    def maybe_valid_and_save(self):
        self.counter += 1
        need_valid_and_update = self.counter % 100 == 0
        if need_valid_and_update: # valid然后更新max_valid_score和last_valid_score，如果分数更大，保存checkpoint
            self.last_valid_score = batch_valid(self.actor.bert, save_readable = False)
            if self.last_valid_score > self.max_valid_score:
                self.max_valid_score = self.last_valid_score
                self.save_checkpoint()
        self.valid_scores.append(self.last_valid_score)
        if need_valid_and_update: # 绘制中间图像
            draw_line_chart(list(range(len(self.episode_rewards))), [self.actor_losses, self.critic_losses, self.valid_scores], ['a', 'c', 'r'])
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
    def traing_and_save(self):
        self.init_params_for_log()
        from ftwp_info import all_train_game_paths
        from bert_for_ftwp import Game_for_rl
        paths = all_train_game_paths()
        game_count = len(paths)
        for index, path in enumerate(paths):
            print(f'{index}/{game_count}')
            game = Game_for_rl(path)
            r,a,c =qac_one_episode(game, self.actor, self.critic, training=True)
            self.episode_rewards.append(r)
            self.actor_losses.append(a)
            self.critic_losses.append(c)
            self.maybe_valid_and_save()
    