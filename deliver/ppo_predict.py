#!/usr/bin/env python
# coding=utf-8
import os
import torch
from ppo import PolicyNet
from env.DeliverEnv import DeliverEnv

action_map = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
state_dim = 2
action_dim = 4
# agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
#             epochs, eps, gamma, device)
model = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

model_path = 'data/ppo.pth'

model.load_state_dict(torch.load(model_path))
# model = torch.load(model_path)
model.eval()

env_name = "deliver"
env = DeliverEnv()
env.reset()

state = env.state

done = False
while not done:
    state = torch.tensor([state], dtype=torch.float)
    probs = model(state)
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample().item()
    actual_action = action_map[action]
    state, reward, done, _ = env.step(actual_action)
    env.render("human")
