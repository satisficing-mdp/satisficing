import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPQFunction(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n
        # self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)  # This is for DDPG style one Q-value-output!
        self.q = mlp([obs_dim] + list(hidden_sizes) + [self.act_dim], activation)

    def forward(self, obs):
        q_values = self.q(obs)
        return torch.squeeze(q_values, -1)  # TODO: Critical to ensure q has right shape. (<--- for DDPG, same for DQN?? Check both cases!)

    def act(self, obs, deterministic=False, epsilon=0.0, *args):
        if not deterministic and np.random.uniform() < epsilon:
            # print("bla")
            a = np.random.choice(self.act_dim)
        else:
            with torch.no_grad():
                q_values = self.forward(obs)
                a = q_values.argmax().numpy()
        return a

    def satisfice(self, obs, epsilon_greedy=True, epsilon=1.0, xi=np.inf):
        # Allows eps_greedy (non-default behaviour!) for easier comparison between satisficing and eps-greedy.
        if epsilon_greedy:
            if np.random.uniform() < epsilon:
                action = np.random.choice(self.act_dim)
                q_values = None
                num_actions = 0
            else:
                with torch.no_grad():
                    q_values = self.forward(obs)
                    action = q_values.argmax().numpy()
                num_actions = len(q_values)
        else:
            # This is real satisficing from here on.
            with torch.no_grad():
                q_values = self.forward(obs)
            satisfying_action_found = False
            shuffled_action_indices = np.random.permutation(self.act_dim)
            action_ix = 0
            while not satisfying_action_found and action_ix < self.act_dim:
                action = shuffled_action_indices[action_ix]
                q_value_ix = q_values[action]
                if q_value_ix >= xi:
                    satisfying_action_found = True
                action_ix += 1
            if not satisfying_action_found:
                # Allow eps-greedy even after all actions have been considered.
                # Otherwise there is no exploration if aspiration level is way too high.
                if np.random.uniform() < epsilon:
                    action = np.random.choice(self.act_dim)
                else:
                    action = q_values.argmax().numpy()
            num_actions = action_ix

        # Return `num_actions` for measuring the number of evaluated actions and `q_values` for the
        return action, num_actions, q_values


class DuelingMLPQFunction(MLPQFunction):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,), activation=nn.ReLU):
        super().__init__(observation_space, action_space, hidden_sizes, activation)
        obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n
        assert len(hidden_sizes) > 1, "The `Dueling network` architecture implemented here needs at least 2 hidden layers (first is for embedding)"

        self.embedding = mlp([obs_dim] + list(hidden_sizes)[:-1], activation=activation, output_activation=activation)
        self.state_val = mlp(list(hidden_sizes)[-2:] + [1], activation=activation)
        self.advantages = mlp(list(hidden_sizes)[-2:] + [self.act_dim], activation=activation)

    def forward(self, obs):
        embedding = self.embedding(obs)
        state_val = self.state_val(embedding)
        advantages = self.advantages(embedding)
        return torch.squeeze(state_val + advantages - advantages.mean(), -1)  # TODO: Critical to ensure q has right shape. (<--- for DDPG, same for DQN?? Check both cases!)

    # def act(self, obs, deterministic=False, epsilon=0.0):
    #     if not deterministic and np.random.uniform() < epsilon:
    #         a = np.random.choice(self.act_dim)
    #     else:
    #         with torch.no_grad():
    #             q_values = self.forward(obs)
    #             a = q_values.argmax().numpy()
    #     return a


class Schedule(object):
    def value(self, step):
        """
        FROM the STABLE-BASELINES repo: https://github.com/hill-a/stable-baselines

        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


class LinearSchedule(Schedule):
    """
    FROM the STABLE-BASELINES repo: https://github.com/hill-a/stable-baselines

    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


"""
Instances of xi_update_fn
"""


def value_tracking(reward, gamma=1, q_value=0, *args, **kwargs):
    return q_value - gamma * reward


def no_negative_value_tracking(reward, gamma=1, q_value=0, *args, **kwargs):
    return q_value - gamma * np.minimum(0, reward)


def aspiration_tracking(xi, reward, *args, **kwargs):
    return xi - reward


def id_update(xi, *args, **kwargs):
    return xi


def aspiration_tracking_discounted(xi, reward, t=1, gamma=1, *args, **kwargs):
    return (xi / (gamma ** t) - (gamma ** t) * reward) * gamma ** (t + 1)


"""
Instances of start_xi_fn
"""


def use_start_xi(start_xi, *args, **kwargs):
    return start_xi


def use_max_q_value(q_values, *args, **kwargs):
    return np.max(q_values)

