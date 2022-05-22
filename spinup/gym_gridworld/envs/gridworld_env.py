import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


def count_dirs(base_rep):
    n, e, s, w = 0, 0, 0, 0
    for ix in base_rep:
        if ix == 0:
            n += 1
        elif ix == 1:
            e += 1
        elif ix == 2:
            s += 1
        elif ix == 3:
            w += 1
    return n, e, s, w


## Find out action numbers for presentations :-)
# comp = {"n": 0, "e": 1, "s": 2, "w": 3}
#
# def action_int(dir_string):
#     expos = 4 ** np.arange(4)
#     su = 0
#     for i in range(len(dir_string)):
#         su += expos[i] * comp[dir_string[i]]
#     return su
#
# action_int("neee")
# action_int("eeen")
# action_int("wnnw")


class FourActionWorldEnv(gym.Env):
    """
    GridworldEnv

    Do not use fires (or "bombs")! They are not implemented properly because action plans
    just "beam" the agent to the target position without caring about intermediate steps.

    Each action (integer) is transformed into a binary number of length 'steps_per_action',
    where 0 means "north" and 1 means "east"
    (e.g., 001010 = 12 = n -> n -> e -> n -> e -> n = 4 up and 2 to the right)

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_rows=8, num_cols=8, steps_per_action=4, predefined_layout=None):
        super(FourActionWorldEnv, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states = self.num_rows * self.num_cols

        self.steps_per_action = steps_per_action
        self.num_actions = int(4 ** self.steps_per_action)
        # self.compass = {0: np.array([1, 0]), 1: np.array([0, 1])}  # , 2: [-1, 0], 3: [0, -1]

        self.observation_space = spaces.Discrete(n=self.num_states)
        self.observation_space_per_action = spaces.Discrete(n=self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)
        self.total_action_space = spaces.Discrete(self.num_actions)

        '''  -- -- Create grids -- -- 
        self.plot_grid is mainly for env.render(). It plots an "A" for the agent.
        self.true_grid does not care about the agent (which is also available from self.agent_position),
                       instead, it is for reward computation and game progress (fires stay fires and are
                       not overridden by the agent "A")  
        '''
        self.plot_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.true_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        self.agent_position = np.array([0, 0])
        self.terminal_states = None
        self.last_pos_was_fire = False
        self.reward_scale_factor = 1
        self.rewards = {" ": -steps_per_action / self.reward_scale_factor,  # Transition into empty cell
                        "A": -steps_per_action / self.reward_scale_factor,  # Hitting the wall (staying on the same cell)
                        "g": (self.num_rows + self.num_cols - steps_per_action) / self.reward_scale_factor,  # Gold
                        "f": -5 / self.reward_scale_factor}  # Fires
        self.done = False
        self.predefined_layout = predefined_layout
        self.reset()

    def step(self, action):
        """
        Parameters
        ----------
        action : integer 0 <= action <= 3 corresponding to the 4 cardinal directions.
        """
        assert isinstance(action, int) and action < self.num_actions
        # Convert into base (4) representation
        base_rep = np.fromstring(np.base_repr(action, base=4).zfill(self.steps_per_action), dtype='S1').astype(int)
        n, e, s, w = count_dirs(base_rep)

        old_position = self.agent_position.copy()
        new_position = old_position.copy()
        new_position[0] = np.clip(new_position[0] + n - s, 0, self.num_rows - 1)
        new_position[1] = np.clip(new_position[1] + e - w, 0, self.num_cols - 1)

        agent_has_moved = np.any(old_position != new_position)
        new_cell = self.true_grid[new_position[0], new_position[1]]
        reward = self.rewards[new_cell]
        if agent_has_moved:
            # Keep grids up to date (mostly for rendering!)
            if self.last_pos_was_fire:
                # Fire is still burning even if leaving the field.
                self.plot_grid[old_position[0], old_position[1]] = "f"
                self.last_pos_was_fire = False
            else:
                self.plot_grid[old_position[0], old_position[1]] = " "
            if new_cell == "g":
                self.true_grid[new_position[0], new_position[1]] = " "
                self.num_remaining_gold -= 1
            elif new_cell == "f":
                self.last_pos_was_fire = True
            self.plot_grid[new_position[0], new_position[1]] = "A"
            self.agent_position = new_position
            self.done = self.check_terminal()

        ob = self._get_feature_values()
        return ob, reward, self.done, {}

    def check_terminal(self):
        return self.num_remaining_gold == 0

    def _get_feature_values(self):
        y, x = self.agent_position
        state = y * self.num_cols + x
        return state

    def reset(self):
        if self.predefined_layout is None:
            self.plot_grid = np.full(shape=(self.num_rows, self.num_cols), fill_value=" ")
            self.plot_grid[self.num_rows - 1, self.num_cols - 1] = "g"
            self.plot_grid[0, 0] = "A"
        # elif self.predefined_layout == "simple":
        #     self.plot_grid = np.flipud(np.array([["g", "f", " ", " ", " ", "g", "g"],  # top
        #                                          [" ", " ", " ", " ", "w", " ", "f"],
        #                                          ["g", " ", " ", "f", " ", " ", "g"],
        #                                          [" ", " ", " ", " ", " ", " ", " "],
        #                                          [" ", "f", "w", " ", "f", " ", " "],
        #                                          [" ", " ", " ", " ", "g", "g", "g"],
        #                                          ["f", "A", " ", "g", " ", "f", "f"]]))  # bottom
        #     self.agent_position = np.array([0, 1])
        else:
            raise NotImplementedError

        agent_ixs = np.where(self.plot_grid == "A")
        assert len(agent_ixs[0]) == 1 and len(agent_ixs[1]) == 1, "Need exactly ONE agent position 'A'"
        self.agent_position = np.array(agent_ixs).flatten()

        terminal_ixs = np.where(np.logical_or(self.plot_grid == "g", self.plot_grid == "b"))
        assert len(terminal_ixs[0]) == 1 and len(terminal_ixs[1]) == 1, "Currently only implemented for one terminal state."
        self.terminal_states = np.array(terminal_ixs).flatten()
        self.true_grid = self.plot_grid.copy()
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        assert self.num_remaining_gold > 0
        assert (self.num_rows, self.num_cols) == self.plot_grid.shape
        self.done = False
        self.last_pos_was_fire = False
        ob = self._get_feature_values()
        return ob

    def render(self, mode='human'):
        self._print_plot_grid()

    def _print_plot_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        # print(np.flipud(self.plot_grid))
        print('\n'.join(['\t'.join(["_" if cell == " " else str(cell) for cell in row]) for row in np.flipud(self.plot_grid)]))

    def _print_true_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        print(np.flipud(self.true_grid))

    def close(self):
        pass


class ActionWorldEnv(gym.Env):
    """
    GridworldEnv

    Do not use fires (or "bombs")! They are not implemented properly because action plans
    just "beam" the agent to the target position without caring about intermediate steps.

    Each action (integer) is transformed into a binary number of length 'steps_per_action',
    where 0 means "north" and 1 means "east"
    (e.g., 001010 = 12 = n -> n -> e -> n -> e -> n = 4 up and 2 to the right)

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_rows=50, num_cols=10, steps_per_action=6, predefined_layout=None):
        super(ActionWorldEnv, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states = self.num_rows * self.num_cols

        self.steps_per_action = steps_per_action
        self.num_actions = int(2 ** self.steps_per_action)
        self.compass = {0: np.array([1, 0]), 1: np.array([0, 1])}  # , 2: [-1, 0], 3: [0, -1]

        self.observation_space = spaces.Discrete(n=self.num_states)
        self.observation_space_per_action = spaces.Discrete(n=self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)
        self.total_action_space = spaces.Discrete(self.num_actions)

        '''  -- -- Create grids -- -- 
        self.plot_grid is mainly for env.render(). It plots an "A" for the agent.
        self.true_grid does not care about the agent (which is also available from self.agent_position),
                       instead, it is for reward computation and game progress (fires stay fires and are
                       not overridden by the agent "A")  
        '''
        self.plot_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.true_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        self.agent_position = np.array([0, 0])
        self.last_pos_was_fire = False
        self.reward_scale_factor = 1
        self.rewards = {" ": -1 / self.reward_scale_factor,  # Transition into empty cell
                        "A": -1 / self.reward_scale_factor,  # Hitting the wall (staying on the same cell)
                        "g": (self.num_rows + self.num_cols) / self.reward_scale_factor,  # Gold
                        "f": -5 / self.reward_scale_factor}  # Fires
        self.done = False
        self.predefined_layout = predefined_layout
        self.reset()

    def step(self, action):
        """
        Parameters
        ----------
        action : integer 0 <= action <= 3 corresponding to the 4 cardinal directions.
        """
        assert isinstance(action, int) and action < self.num_actions
        # Convert into binary representation
        ba = np.fromstring(np.binary_repr(action, width=self.steps_per_action), dtype='S1').astype(int)

        # Convert binary representation into n_up and
        n_east = np.sum(ba)
        n_north = self.steps_per_action - n_east

        old_position = self.agent_position.copy()
        new_position = old_position.copy()
        new_position[0] = np.minimum(new_position[0] + n_north, self.num_rows - 1)
        new_position[1] = np.minimum(new_position[1] + n_east, self.num_cols - 1)

        agent_has_moved = np.any(old_position != new_position)
        new_cell = self.true_grid[new_position[0], new_position[1]]
        reward = self.rewards[new_cell]
        if agent_has_moved:
            # Keep grids up to date (mostly for rendering!)
            if self.last_pos_was_fire:
                # Fire is still burning even if leaving the field.
                self.plot_grid[old_position[0], old_position[1]] = "f"
                self.last_pos_was_fire = False
            else:
                self.plot_grid[old_position[0], old_position[1]] = " "
            if new_cell == "g":
                self.true_grid[new_position[0], new_position[1]] = " "
                self.num_remaining_gold -= 1
            elif new_cell == "f":
                self.last_pos_was_fire = True
            self.plot_grid[new_position[0], new_position[1]] = "A"
            self.agent_position = new_position
            self.done = self.check_terminal()

        ob = self._get_feature_values()
        return ob, reward, self.done, {}

    def check_terminal(self):
        return self.num_remaining_gold == 0

    def _get_feature_values(self):
        y, x = self.agent_position
        state = y * self.num_cols + x
        return state

    def reset(self):
        if self.predefined_layout is None:
            self.plot_grid = np.full(shape=(self.num_rows, self.num_cols), fill_value=" ")
            self.plot_grid[self.num_rows - 1, self.num_cols - 1] = "g"
            self.plot_grid[0, 0] = "A"
        # elif self.predefined_layout == "simple":
        #     self.plot_grid = np.flipud(np.array([["g", "f", " ", " ", " ", "g", "g"],  # top
        #                                          [" ", " ", " ", " ", "w", " ", "f"],
        #                                          ["g", " ", " ", "f", " ", " ", "g"],
        #                                          [" ", " ", " ", " ", " ", " ", " "],
        #                                          [" ", "f", "w", " ", "f", " ", " "],
        #                                          [" ", " ", " ", " ", "g", "g", "g"],
        #                                          ["f", "A", " ", "g", " ", "f", "f"]]))  # bottom
        #     self.agent_position = np.array([0, 1])
        else:
            raise NotImplementedError

        agent_ixs = np.where(self.plot_grid == "A")
        assert len(agent_ixs[0]) == 1 and len(agent_ixs[1]) == 1, "Need exactly ONE agent position 'A'"
        self.agent_position = np.array(agent_ixs).flatten()
        self.true_grid = self.plot_grid.copy()
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        assert self.num_remaining_gold > 0
        assert (self.num_rows, self.num_cols) == self.plot_grid.shape
        self.done = False
        self.last_pos_was_fire = False
        ob = self._get_feature_values()
        return ob

    def render(self, mode='human'):
        self._print_plot_grid()

    def _print_plot_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        # print(np.flipud(self.plot_grid))
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in np.flipud(self.plot_grid)]))

    def _print_true_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        print(np.flipud(self.true_grid))

    def close(self):
        pass


class GridworldEnv(gym.Env):
    """
    GridworldEnv

    At the moment, fires (or bombs) are NOT terminal states; only give negative rewards.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_rows=5, num_cols=5, predefined_layout=None):
        super(GridworldEnv, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states = self.num_rows * self.num_cols

        # self.stochasticity_eps = 0
        self.action_names = {0: "n", 1: "e", 2: "s", 3: "w"}
        self.num_actions = len(self.action_names)
        self.compass = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}

        self.observation_space = spaces.Discrete(n=self.num_states)
        self.observation_space_per_action = spaces.Discrete(n=self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)
        self.total_action_space = spaces.Discrete(self.num_actions)

        '''  -- -- Create grids -- -- 
        self.plot_grid is mainly for env.render(). It plots an "A" for the agent.
        self.true_grid does not care about the agent (which is also available from self.agent_position),
                       instead, it is for reward computation and game progress (fires stay fires and are
                       not overridden by the agent "A")  
        '''
        self.plot_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.true_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        self.agent_position = np.array([0, 0])
        self.last_pos_was_fire = False
        self.reward_scale_factor = 1
        self.rewards = {" ": -1 / self.reward_scale_factor,  # Transition into empty cell
                        "A": -1 / self.reward_scale_factor,  # Hitting the wall (staying on the same cell)
                        "g": 10 / self.reward_scale_factor,  # Gold
                        "f": -5 / self.reward_scale_factor}  # Fires
        self.done = False
        self.predefined_layout = predefined_layout
        self.reset()

    def step(self, action):
        """
        Parameters
        ----------
        action : integer 0 <= action <= 3 corresponding to the 4 cardinal directions.
        """
        assert action in self.action_names.keys()
        # if np.random.uniform(0, 1) < self.stochasticity_eps:
        #     action = np.random.choice(GridworldEnv.action_names.keys())

        old_position = self.agent_position.copy()
        new_position = self.agent_position.copy()
        candidate_position = self.agent_position.copy() + self.compass[action]
        if (0 <= candidate_position[0] < self.num_rows and
                0 <= candidate_position[1] < self.num_cols and
                self.plot_grid[candidate_position[0], candidate_position[1]] != "w"):
            new_position = candidate_position

        agent_has_moved = np.any(old_position != new_position)
        new_cell = self.true_grid[new_position[0], new_position[1]]
        reward = self.rewards[new_cell]
        if agent_has_moved:
            # new_cell = self.true_grid[new_position[0], new_position[1]]
            # reward = self.rewards[new_cell]
            if self.last_pos_was_fire:
                # Fire is still burning even if leaving the field.
                self.plot_grid[old_position[0], old_position[1]] = "f"
                self.last_pos_was_fire = False
            else:
                self.plot_grid[old_position[0], old_position[1]] = " "
            if new_cell == "g":
                self.true_grid[new_position[0], new_position[1]] = " "
                self.num_remaining_gold -= 1
            elif new_cell == "f":
                self.last_pos_was_fire = True
            self.plot_grid[new_position[0], new_position[1]] = "A"
            self.agent_position = new_position
            self.done = self.check_terminal()

        ob = self._get_feature_values()
        return ob, reward, self.done, {}

    def check_terminal(self):
        return self.num_remaining_gold == 0

    def _get_feature_values(self):
        y, x = self.agent_position
        state = y * self.num_cols + x
        return state

    def reset(self):
        if self.predefined_layout is None:
            self.plot_grid = np.full(shape=(self.num_rows, self.num_cols), fill_value=" ")
            self.plot_grid[self.num_rows - 1, self.num_cols - 1] = "g"
            self.plot_grid[1, 1] = "A"
        # elif self.predefined_layout == "simple":
        #     self.plot_grid = np.flipud(np.array([["g", "f", " ", " ", " ", "g", "g"],  # top
        #                                          [" ", " ", " ", " ", "w", " ", "f"],
        #                                          ["g", " ", " ", "f", " ", " ", "g"],
        #                                          [" ", " ", " ", " ", " ", " ", " "],
        #                                          [" ", "f", "w", " ", "f", " ", " "],
        #                                          [" ", " ", " ", " ", "g", "g", "g"],
        #                                          ["f", "A", " ", "g", " ", "f", "f"]]))  # bottom
        #     self.agent_position = np.array([0, 1])
        else:
            raise NotImplementedError

        agent_ixs = np.where(self.plot_grid == "A")
        assert len(agent_ixs[0]) == 1 and len(agent_ixs[1]) == 1, "Need exactly ONE agent position 'A'"
        self.agent_position = np.array(agent_ixs).flatten()
        self.true_grid = self.plot_grid.copy()
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        assert self.num_remaining_gold > 0
        assert (self.num_rows, self.num_rows) == self.plot_grid.shape
        self.done = False
        self.last_pos_was_fire = False
        ob = self._get_feature_values()
        return ob

    def render(self, mode='human'):
        self._print_plot_grid()

    def _print_plot_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        print(np.flipud(self.plot_grid))

    def _print_true_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        print(np.flipud(self.true_grid))

    def close(self):
        pass


