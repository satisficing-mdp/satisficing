import numpy as np
from gym.spaces import Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def id_update(xi, *args, **kwargs):
    return xi


def id_update_start(start_xi, *args, **kwargs):
    return start_xi


def decrease_by_lr_times_reward(xi, reward, lr,  *args, **kwargs):
    return xi - lr * reward


def decrease_by_reward(xi, reward, *args, **kwargs):
    return xi - reward


def increase_by_q_dif(xi, q_t_minus_one, q_t_minus_two, *args, **kwargs):
    return xi + q_t_minus_one - q_t_minus_two


def set_to_last_return(start_xi, episode_return, lr, *args, **kwargs):
    return episode_return


def set_to_lr_times_return_div_length(start_xi, episode_return, lr, episode_length, *args, **kwargs):
    return lr * episode_return / episode_length


def set_to_lr_times_return(start_xi, episode_return, lr, *args, **kwargs):
    return lr * episode_return


def increase_by_return(start_xi, episode_return, lr, *args, **kwargs):
    return start_xi + episode_return


def increase_by_lr_times_return(start_xi, episode_return, lr, *args, **kwargs):
    return start_xi + lr * (episode_return - start_xi)


def increase_by_lrsq_times_return(start_xi, episode_return, lr, *args, **kwargs):
    return start_xi + (lr**2) * (episode_return - start_xi)


class TabSARSA:
    """
    'xi' only for compatibility reasons with

    """
    def __init__(self,
                 observation_space,
                 action_space,
                 epsilon,
                 alpha,
                 xi=0,
                 xi_update_fn=id_update,
                 start_xi_update_fn=id_update,
                 gamma=1):
        assert isinstance(action_space, Discrete)
        self.num_actions = action_space.n
        self.q_table = np.zeros(shape=(observation_space.n, self.num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.start_xi = xi
        self.xi_update_fn = xi_update_fn
        self.start_xi_update_fn = start_xi_update_fn
        self.gamma = gamma

    def choose_action(self, obs, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.num_actions))
            num_actions = 1
        else:
            q_vals = self.q_table[obs, :]
            max_q = np.max(q_vals)
            action = np.random.choice(np.arange(self.num_actions)[q_vals == max_q])
            num_actions = self.num_actions
        return action, num_actions

    def learn(self, obs, act, next_obs, next_act, r):
        self.q_table[obs, act] += self.alpha * (r + self.gamma * self.q_table[next_obs, next_act] - self.q_table[obs, act])

    def reset(self):
        self.reset_xi()

    def reset_xi(self):
        self.xi = self.start_xi

    def update_xi(self, reward, q_t_minus_one=None, q_t_minus_two=None):
        new_xi = self.xi_update_fn(xi=self.xi, reward=reward, lr=self.alpha, q_t_minus_one=q_t_minus_one, q_t_minus_two=q_t_minus_two)
        self.xi = new_xi

    def update_start_xi(self, episode_return, episode_length):
        self.start_xi = self.start_xi_update_fn(start_xi=self.start_xi, episode_return=episode_return, lr=self.alpha, episode_length=episode_length)


class SatTabSARSA(TabSARSA):
    def __init__(self,
                 observation_space,
                 action_space,
                 epsilon,
                 alpha,
                 xi=0,
                 xi_update_fn=id_update,
                 start_xi_update_fn=id_update,
                 gamma=1):
        super().__init__(observation_space, action_space, epsilon, alpha, xi, xi_update_fn, start_xi_update_fn, gamma)

    def choose_action(self, obs, greedy=False):
        if greedy:
            q_vals = self.q_table[obs, :]
            max_q = np.max(q_vals)
            action = np.random.choice(np.arange(self.num_actions)[q_vals == max_q])
            num_actions = self.num_actions
        else:
            satisfying_action_found = False
            actions = np.random.permutation(self.num_actions)
            q_values = np.zeros(self.num_actions)
            action_ix = 0
            while not satisfying_action_found and action_ix < self.num_actions:
                action = actions[action_ix]
                q_value_ix = self.q_table[obs, action]
                q_values[action] = q_value_ix
                if q_value_ix >= self.xi:
                    satisfying_action_found = True
                action_ix += 1
            if not satisfying_action_found:
                if np.random.random() < self.epsilon:
                    action = np.random.choice(np.arange(self.num_actions))
                else:
                    max_q = np.max(q_values)
                    action = np.random.choice(np.arange(self.num_actions)[q_values == max_q])
            num_actions = action_ix
        return action, num_actions

    def learn(self, obs, act, next_obs, next_act, r):
        self.q_table[obs, act] += self.alpha * (r + self.gamma * self.q_table[next_obs, next_act] - self.q_table[obs, act])
        self.update_xi(reward=r, q_t_minus_one=self.q_table[next_obs, next_act], q_t_minus_two=self.q_table[obs, act])


class RandomAgent(TabSARSA):
    def __init__(self, observation_space, action_space, epsilon, alpha, xi=0, xi_update_fn=id_update, gamma=1):
        super().__init__(observation_space, action_space, epsilon, alpha, xi, xi_update_fn, gamma)

    def choose_action(self, obs, greedy=False):
        return np.random.choice(np.arange(self.num_actions)), 1

    def learn(self, obs, act, next_obs, next_act, r):
        pass


class TabQLearning(TabSARSA):
    def __init__(self,
                 observation_space,
                 action_space,
                 epsilon,
                 alpha,
                 xi=0,
                 xi_update_fn=id_update,
                 start_xi_update_fn=id_update,
                 gamma=1):
        super().__init__(observation_space, action_space, epsilon, alpha, xi, xi_update_fn, start_xi_update_fn, gamma)

    def learn(self, obs, act, next_obs, next_act, r):
        max_next_q_value = np.max(self.q_table[next_obs, :])
        self.q_table[obs, act] += self.alpha * (r + self.gamma * max_next_q_value - self.q_table[obs, act])

    def choose_action(self, obs, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.num_actions))
            # difference to SARSA... because learning takes max over all Q values, need to evaluate all of them.
            # It's just that num_actions is counted here rather than in the learning step
            num_actions = self.num_actions
        else:
            q_vals = self.q_table[obs, :]
            max_q = np.max(q_vals)
            action = np.random.choice(np.arange(self.num_actions)[q_vals == max_q])
            num_actions = self.num_actions
        return action, num_actions


class TabQ:
    """
    Used this version for something...
    Please use TabQlearning now.

    """
    def __init__(self,
                 observation_space,
                 action_space,
                 epsilon,
                 alpha,
                 xi=0,
                 xi_update_fn=id_update,
                 gamma=1):
        assert isinstance(action_space, Discrete)
        self.num_actions = action_space.n
        self.q_table = np.zeros(shape=(observation_space.n, self.num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.start_xi = xi
        self.xi_update_fn = xi_update_fn
        self.gamma = gamma

    def choose_action(self, obs, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.num_actions))
            num_actions = 1
        else:
            q_vals = self.q_table[obs, :]
            max_q = np.max(q_vals)
            action = np.random.choice(np.arange(self.num_actions)[q_vals == max_q])
            num_actions = self.num_actions
        return action, num_actions

    def learn(self, obs, act, next_obs, next_act, r):
        max_next_q_value = np.max(self.q_table[next_obs, :])
        self.q_table[obs, act] += self.alpha * (r + self.gamma * max_next_q_value - self.q_table[obs, act])

    def reset(self):
        self.reset_xi()

    def reset_xi(self):
        self.xi = self.start_xi

    def update_xi(self, reward):
        self.xi = self.xi_update_fn(self.xi, reward, self.alpha)
