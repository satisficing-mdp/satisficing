from gym.envs.registration import register

register(
    id='Gridworld-v0',
    entry_point='spinup.gym_gridworld.envs:GridworldEnv',
    max_episode_steps=500,
)

register(
    id='ActionWorld-v0',
    entry_point='spinup.gym_gridworld.envs:ActionWorldEnv',
    max_episode_steps=500,
)

register(
    id='FourActionWorld-v0',
    entry_point='spinup.gym_gridworld.envs:FourActionWorldEnv',
    max_episode_steps=1000,
)
