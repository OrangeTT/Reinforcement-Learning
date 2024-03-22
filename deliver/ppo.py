#!/usr/bin/env python
# coding=utf-8
import os
import sys
# print(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gym
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import rl_utils
import random
from env.DeliverEnv import DeliverEnv

action_map = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x/5
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x/5
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, random_action, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.random_action = random_action
        self.action_dim = action_dim

    def take_action(self, state):
        if np.random.random() < self.random_action:
            action = np.random.randint(self.action_dim)
            return action
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            # print(state)
            probs = self.actor(state)
            # print(probs)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == '__main__':
    actor_lr = 1e-5
    critic_lr = 1e-4
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 50
    eps = 0.2
    random_action = 0.1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    # env_name = 'CartPole-v0'
    # env = gym.make(env_name)
    # env.seed(0)
    # torch.manual_seed(0)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    env_name = "deliver"
    env = DeliverEnv()
    # env.reset()

    torch.manual_seed(0)
    state_dim = 2
    action_dim = 4
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, random_action, device)

    save_dir = 'data'

    return_list = []
    for i in range(epochs):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                env.reset()
                state = env.state
                done = False
                while not done:
                    # print(state)
                    action = agent.take_action(state)
                    actual_action = action_map[action]
                    next_state, reward, done, _ = env.step(actual_action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    # env.render(mode='human')
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                # env.close()
        if i == epochs - 1:
            torch.save(agent.actor.state_dict(), os.path.join(save_dir, 'ppo.pth'))
            # torch.save(agent.actor, os.path.join(save_dir, 'ppo.pth'))

    # env.close()

