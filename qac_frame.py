import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # 输出概率分布（离散动作）
        )
    
    def forward(self, state):
        return self.fc(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Q(s, a) 值
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

class QACAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state).detach().numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)  # 采样动作
        return action
    
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action_onehot = torch.zeros(1, action_dim)  # 这样更加通用
        action_onehot[0, action] = 1.0
        
        # 计算目标 Q 值 (TD 目标)
        with torch.no_grad():
            next_action_probs = self.actor(next_state)
            next_action = torch.argmax(next_action_probs, dim=-1)
            next_action_onehot = torch.zeros_like(next_action_probs)
            next_action_onehot[0, next_action] = 1.0
            target_q = reward + (1 - done) * 0.99 * self.critic(next_state, next_action_onehot)
        
        # 计算当前 Q 值
        current_q = self.critic(state, action_onehot)
        
        # 更新 Critic
        critic_loss = self.criterion(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新 Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# 训练循环
def train_loop(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# 示例: 初始化 QAC 代理并训练
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = QACAgent(state_dim, action_dim)
train_loop(env, agent)