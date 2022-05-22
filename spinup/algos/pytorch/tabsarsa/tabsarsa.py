import numpy as np
# Needed to register Gym environments.
import gym
import spinup.gym_gridworld
import time
import spinup.algos.pytorch.tabsarsa.core as core
from spinup.algos.pytorch.tabsarsa.core import id_update, decrease_by_reward
from spinup.utils.logx import EpochLogger


def test_xis_for_q_star(env_fn,
                        actor=core.SatTabSARSA,
                        seed=None,
                        steps_per_epoch=4000,
                        epochs=50,
                        gamma=1,
                        epsilon=0.0,
                        vf_lr=1e-3,
                        xi=0,
                        xi_update_fn=id_update,
                        max_ep_len=50,
                        num_eval_runs=10,
                        eval_greedy=False,
                        logger_kwargs=dict(),
                        save_freq=10,
                        # mask_invalid_actions=True,
                        verbose=False,
                        random_eval_env=False):

    if isinstance(actor, str):
        actor = eval(actor)

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed = int(seed * 30)
    # Instantiate environment
    if seed is not None:
        np.random.seed(seed)
    env = env_fn()

    if seed is not None:
        np.random.seed(seed)
    eval_env = env_fn()

    if seed is not None:
        # torch.manual_seed(seed)
        np.random.seed(seed)

    eval_ac = actor(observation_space=env.observation_space,
                    action_space=env.action_space,
                    epsilon=epsilon,
                    alpha=vf_lr,
                    xi=xi,
                    xi_update_fn=xi_update_fn,
                    gamma=gamma)

    q_star = np.load("pretrained_models/MacroGridworld-QTables/q_star_value_iteration.npy")
    assert np.all(eval_ac.q_table.shape == q_star.shape)
    print(f"Min Q-value={np.min(q_star[:-1])}")
    print(f"Max Q-value={np.max(q_star)}")
    print(f"Mean Q-value={np.mean(q_star)}")
    eval_ac.q_table = q_star

    def evaluate(eval_env):
        if random_eval_env:
            eval_env = env_fn()
        for run in range(num_eval_runs):
            eval_o = eval_env.reset()
            eval_ac.reset_xi()
            eval_a, eval_num_actions = eval_ac.choose_action(obs=eval_o, greedy=eval_greedy)
            eval_ret = 0
            eval_d = False
            eval_len = 0
            eval_sum_actions = eval_num_actions
            while not eval_d and eval_len < max_ep_len:
                eval_next_o, eval_r, eval_d, eval_info = eval_env.step(int(eval_a))
                eval_ret += eval_r

                eval_ac.update_xi(reward=eval_r)

                eval_next_a, eval_num_actions = eval_ac.choose_action(obs=eval_next_o, greedy=eval_greedy)
                eval_sum_actions += eval_num_actions

                eval_a = eval_next_a
                eval_o = eval_next_o

                eval_len += 1

            logger.store(EpRet=eval_ret)
            logger.store(EpSumAct=eval_sum_actions)
            logger.store(EpAvgAct=eval_sum_actions/eval_len)

        logger.log_tabular('EpRet')  # , with_min_and_max=True
        logger.log_tabular('EpSumAct')  # , with_min_and_max=True
        logger.log_tabular('EpAvgAct')  # , with_min_and_max=True
        logger.log_tabular('xi', xi)
        logger.dump_tabular()

    # Prepare for interaction with environment
    start_time = time.time()
    evaluate(eval_env)


