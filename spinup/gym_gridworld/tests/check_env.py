import numpy as np
import gym
# import gym_gridworld  # needed to register
import time
# from stable_baselines.common.env_checker import check_env  # to check provided feature_gridworld env

env = gym.make('FourActionWorld-v0')  # , num_rows=10, num_cols=10, steps_per_action=4
env.reset()
env.render()
env.step(100)
env.render()
# env.step(15)
# env.step(1)
d = False
while not d:
    o, r, d, _ = env.step(np.random.choice(env.action_space.n))
    env.render()
    time.sleep(0.2)


# env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
#
# # (From stable-baselines:) It will check your custom environment and output additional warnings if needed
# check_env(env)