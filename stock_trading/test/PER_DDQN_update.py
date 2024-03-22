
import os
import copy
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from highway_env.envs import IntersectionEnv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env

PATH= 'save'

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("choose device...",device)


# Define the environment
env = gym.make('intersection-v0',render_mode='rgb_array')
#env = gym.make("merge-v0",render_mode='rgb_array')
env.reset()
# details
env.config["duration"] = 13
env.config["vehicles_count"] = 20
env.config["vehicles_density"] = 1.3
env.config["reward_speed_range"] = [7.0, 10.0]
env.config["initial_vehicle_count"] = 10
env.config["simulation_frequency"] = 15
env.config["arrived_reward"] = 4
env.reset()

#十字路口环境的结构
env.config
{
    "observation": {
        "type": "Kinematics",#Kinematics
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": True,
        "flatten": False,
        "observe_intentions": False
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": False,
        "lateral": True
    },
    "duration": 13,  # [s]
    "destination": "o1",
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "collision_reward": -3,
    "normalize_reward": False
}


class ReplayBuffer:

    def __init__(self, capacity, alpha, beta):
        """
        ### Initialize
        """

        self.capacity = capacity

        self.alpha = alpha
        #alpha 决定了优先级的权重，较大的 alpha 会使得优先级的差异更为显著。
        self.beta = beta
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1.

        # Arrays for buffer
        self.data = {
            'obs': np.zeros(shape=(capacity, 15, 7)),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, 15, 7)),
            'done': np.zeros(shape=capacity, dtype=bool)
        }
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx


        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        idx = 1
        while idx < self.capacity:

            if self.priority_sum[idx * 2] > prefix_sum:

                idx = 2 * idx
            else:

                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1


        return idx - self.capacity

    def sample(self, batch_size):
        """
        ### Sample from buffer
        """

        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
        prob_min = self._min() / self._sum()

        max_weight = (prob_min * self.size) ** (-self.beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]

            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-self.beta)
            samples['weights'][i] = weight / max_weight

        # Get samples data
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples
    def calculate_priority(self, td_error):
        """
        ### Calculate priority for a given TD error
        """
        return (abs(td_error) + 1e-6) ** self.alpha
    def calculate_priority_add(self, td_error,average_eps_reward):
        """
        ### Calculate priority for a given TD error
        """
        return (abs(td_error) + average_eps_reward) ** self.alpha

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)

            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        # super class
        super(Net, self).__init__()
        # hidden nodes define
        hidden_nodes1 = 1024#1024
        hidden_nodes2 = 512
        self.fc1 = nn.Linear(state_dim, hidden_nodes1)
        self.fc2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.fc3 = nn.Linear(hidden_nodes2, action_dim)

    def forward(self, state):
        # define forward pass of the actor
        x = state # state
        # Relu function double
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DOUBLEDQN(nn.Module):
    def __init__(
        self,
            env,
            buffer,# gym environment
            state_dim, # state size
            action_dim, # action size

        lr = 0.001, # learning rate
        gamma = 0.99, # discount factor
        batch_size = 5, # batch size for each training
        timestamp = "",):

        # super class
        super(DOUBLEDQN, self).__init__()
        self.env = env
        self.env.reset()
        self.timestamp = timestamp
        # for evaluation purpose
        self.test_env = copy.deepcopy(env)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.is_rend = False
        self.target_net = Net(self.state_dim, self.action_dim).to(device)#TODO
        self.estimate_net = Net(self.state_dim, self.action_dim).to(device)#TODO

        self.optimizer = torch.optim.Adam(self.estimate_net.parameters(), lr=lr)
        self.replaybuffer = buffer
    def choose_our_action(self, state, epsilon = 0.99):
        # greedy strategy for choosing action
        # state: ndarray environment state
        # epsilon: float in [0,1]
        # return: action we chosen
        # turn to 1D float tensor -> [[a1,a2,a3,...,an]]
        # we have to increase the speed of transformation ndarray to tensor if not it will spend a long time to train the model
        # ndarray[[ndarray],...[ndarray]] => list[[ndarray],...[ndarray]] => ndarray[...] => tensor[...]
        if type(state) == type((1,)):
            state = state[0]
        temp = [exp for exp in state]
        target = []
        target = np.array(target)
        # n dimension to 1 dimension ndarray
        for i in temp:
            target = np.append(target,i)
        state = torch.FloatTensor(target).to(device)
        # randn() return a set of samples which are Gaussian distribution
        # no argments -> return a float number
        if np.random.randn() <= epsilon:
            # when random number smaller than epsilon: do these things
            # put a state array into estimate net to obtain their value array
            # choose max values in value array -> obtain action
            action_value = self.estimate_net(state)
            action = torch.argmax(action_value).item()
        else:
            # when random number bigger than epsilon: randomly choose a action
            action = np.random.randint(0, self.action_dim)

        return action

    def train(self, num_episode):
        # num_eposide: total turn number for train
        loss_list = [] # loss set
        avg_reward_list = [] # reward set

        rend = 0

        # tqdm : a model for showing process bar
        for episode in tqdm(range(1,int(num_episode)+1)):
            done = False
            state = self.env.reset()
            each_loss = 0
            step = 0
            episode_total_reward = 0
            episode_average_loss = 0
            if type(state) == type((1,)):
                state = state[0]
            while not done:
                if self.is_rend:
                    self.env.render()

                step +=1
                action = self.choose_our_action(state)
                observation, reward, done, truncated, info = self.env.step(action)
                #env.render()
                exp = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "state_next": observation,
                    "done": done,
                }
                self.replaybuffer.add(exp["state"],exp["action"],exp["reward"],exp["state_next"],exp["done"])


                state = observation

                # sample random batch in replay memory
                samples = self.replaybuffer.sample(self.batch_size)
                weights_data = samples['weights']  # 权重数据
                indexes_data = samples['indexes']  # 索引数据
                obs_data = samples['obs']  # 观察数据
                action_data = samples['action']  # 动作数据
                reward_data = samples['reward']  # 奖励数据
                next_obs_data = samples['next_obs']  # 下一个观察数据
                done_data = samples['done']  # 完成标志数据

                done_data_inverted = ~done_data

                # extract batch data
                action_batch = torch.LongTensor(action_data).to(device)
                reward_batch = torch.FloatTensor(reward_data).to(device)
                done_batch = torch.FloatTensor(done_data_inverted).to(device)
                indexes_tensor = torch.LongTensor(indexes_data).to(device)  # 转换为 PyTorch 张量
                weights_tensor = torch.FloatTensor(weights_data).to(device)
                # Slow method -> Fast method when having more data

                state_temp_list = np.array(obs_data)
                state_next_temp_list = np.array(next_obs_data)

                state_next_batch = torch.FloatTensor(state_next_temp_list).to(device)
                state_batch = torch.FloatTensor(state_temp_list).to(device)


                # reshape
                state_batch = state_batch.reshape(self.batch_size, -1)
                action_batch = action_batch.reshape(self.batch_size, -1)
                reward_batch = reward_batch.reshape(self.batch_size, -1)
                state_next_batch = state_next_batch.reshape(self.batch_size, -1)
                done_batch = done_batch.reshape(self.batch_size, -1)
                #print("action_batch.shape",action_batch.shape)
                #print("state_batch.shape", state_batch.shape)
                #print("reward_batch.shape", reward_batch.shape)
                #print("state_next_batch.shape", state_next_batch.shape)
                #print("done_batch.shape", done_batch.shape)
                # obtain estimate Q value gather(dim, index) dim==1:column index
                max_action_index = action_batch.max()
                output_shape = self.estimate_net(state_batch).shape
                if max_action_index >= output_shape[1]:
                    print("Action index out of range for estimate_net output")
                #print(f"Max action index: {max_action_index}")
                #print(f"Output shape of estimate_net: {output_shape}")

                estimate_Q_value = self.estimate_net(state_batch).gather(1, action_batch)

                # obtain target Q value detach:remove the matched element
                max_action_index = self.estimate_net(state_next_batch).detach().argmax(1)
                target_Q_value = reward_batch + done_batch * self.gamma * self.target_net(
                    state_next_batch
                ).gather(1, max_action_index.unsqueeze(1))# squeeze(1) n*1->1*1, unsqueeze(1) 1*1->n*1

                loss = (weights_tensor * F.mse_loss(estimate_Q_value, target_Q_value))
                loss_mean = loss.mean()

                #print("loss:",loss)
                #print("reward:",reward)
                each_loss += loss_mean.item()
                episode_total_reward += reward
                epi_r_avg = episode_total_reward / step
                loss_reward_mean = loss_mean
                # update network
                self.optimizer.zero_grad()
                loss_reward_mean.backward()
                self.optimizer.step()
                # 使用损失更新优先级
                new_p_tensor = self.replaybuffer.calculate_priority(loss+epi_r_avg)
                #print("new_p:", new_p_tensor)
                self.replaybuffer.update_priorities(indexes_tensor,new_p_tensor)
                # update target network
                # load parameters into model
                if self.learn_step_counter % 10 == 0:
                    self.target_net.load_state_dict(self.estimate_net.state_dict())
                self.learn_step_counter +=1


            #reward, count = self.eval()
            #episode_reward += reward

            #使用奖励和损失更新优先级
            #new_p_tensor = self.replaybuffer.calculate_priority_add(loss, epi_r_avg)
            # print("new_p:", new_p_tensor)
            #self.replaybuffer.update_priorities(indexes_tensor, new_p_tensor)
            # you can update these variables


            # save
            period = 1
            if episode % period == 0:
                each_loss/= step

                avg_reward_list.append(epi_r_avg*step)
                loss_list.append(each_loss)

                print("\nepisode:[{}/{}], \t eposide_avg_loss: {:.4f}, \t eposide_avg_reward: {:.3f}, \t step: {}".format(
                    episode, num_episode, each_loss, epi_r_avg, step#count
                ))

                # episode_reward = 0
                # create a new directory for saving
                path = PATH + "/" + self.timestamp
                try:
                    os.makedirs(path)
                except OSError:
                    pass
                # saving as timestamp file
                np.save(path + "/DOUBLE_DQN_LOSS.npy", loss_list)
                np.save(path + "/DOUBLE_DQN_EACH_REWARD.npy", avg_reward_list)
                torch.save(self.estimate_net.state_dict(), path + "/DOUBLE_DQN_params.pkl")

        self.env.close()
        return loss_list, avg_reward_list
"""
    def eval(self):
        # evaluate the policy
        count = 0
        total_reward = 0
        done = False
        state = self.test_env.reset()
        if type(state) == type((1,)):
            state = state[0]

        while not done:
            action = self.choose_our_action(state, epsilon = 1)
            observation, reward, done, truncated, info = self.test_env.step(action)
            total_reward += reward
            count += 1
            state = observation

        return total_reward, count
"""
if __name__ == "__main__":
    capacity = 1000
    # alpha 决定了优先级的权重，较大的 alpha 会使得优先级的差异更为显著。
    # 如果beta为0，则所有样本将具有相同的权重，不考虑它们的优先级。这对应于均匀采样。
    # 随着beta的增加，分配给高优先级样本的权重将增加，使它们有更高的被采样的机会。
    alpha = 0.7
    beta = 0.75

    buffer = ReplayBuffer(capacity, alpha,beta)

    # timestamp
    named_tuple = time.localtime()
    time_string = time.strftime("%Y-%m-%d-%H-%M", named_tuple)
    print(time_string)
    # create a doubledqn object
    double_dqn_object = DOUBLEDQN(
        env,
        buffer,
        state_dim=105,#105
        action_dim=3,
        lr=0.001,
        gamma=0.99,
        batch_size=64,
        timestamp=time_string,
    )
    # your chosen train times
    iteration = 50
    # start training
    avg_loss, avg_reward_list = double_dqn_object.train(iteration)
    path = os.path.join('save', time_string)
    np.save(path + "/DOUBLE_DQN_LOSS.npy", avg_loss)
    np.save(path + "/DOUBLE_DQN_EACH_REWARD.npy", avg_reward_list)
    torch.save(double_dqn_object.estimate_net.state_dict(), path + "/DOUBLE_DQN_params.pkl")
    torch.save(double_dqn_object.state_dict(), path + "/DOUBLE_DQN_MODEL.pt")
