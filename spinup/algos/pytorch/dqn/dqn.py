from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.dqn.core as core
from spinup.utils.logx import EpochLogger


class DQNReplayBuffer:
    """
    A simple experience replay buffer for DQN agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) if k != "act" else torch.as_tensor(v, dtype=torch.int64) for k,v in batch.items()}



def dqn(env_fn, q_net=core.MLPQFunction, q_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=100000, gamma=0.99,
        vf_lr=1e-3, alpha=0.00025, batch_size=32, start_steps=0,
        update_after=32, update_every=1, update_target_network_every=100,
        polyak=0.0, double_q_learning=False, exploration_fraction=0.5,
        exploration_final_eps=0.05, exploration_initial_eps=1.0,
        num_test_episodes=20, max_ep_len=None,
        eval_greedy=True,  # epsilon=0.0,
        logger_kwargs=dict(), save_freq=10):
    """
    Deep Q-Network (DQN)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        vf_lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the greedy
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    if max_ep_len is None:
        max_ep_len = env._max_episode_steps
    else:
        env._max_episode_steps = max_ep_len
        test_env._max_episode_steps = max_ep_len
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    # Create q and target networks
    if isinstance(q_net, str):
        q_net = eval(q_net)
    q = q_net(env.observation_space, env.action_space, **q_kwargs)
    q_targ = deepcopy(q)

    # Freeze target networks with respect to optimizers (only update via target updates)
    for p in q_targ.parameters():
        p.requires_grad = False
        
    # # List of parameters for both Q-networks (save this for convenience)
    # q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = DQNReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    logger.log(f'\nNumber of parameters: \t {core.count_vars(q)}')

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q_values = q(o).gather(1, a.unsqueeze(1)).squeeze()
        if double_q_learning:
            with torch.no_grad():
                actions_that_are_qmax_w_r_t_q = q(o2).max(1)[1]
                q_target_values = q_targ(o2).gather(1, actions_that_are_qmax_w_r_t_q.unsqueeze(1)).squeeze()
        else:
            q_target_values = q_targ(o2).max(1)[0]

        bellman_targets = r + gamma * q_target_values * (1-d)
        q_loss = ((q_values - bellman_targets) ** 2).mean()
        return q_loss

    # Set up optimizers for policy and q-function
    # pi_optimizer = Adam(ac.pi.parameters(), lr=vf_lr)
    q_optimizer = Adam(q.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(q)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item())

        # Finally, update target networks by polyak averaging.
        if t % update_target_network_every == 0:
            with torch.no_grad():
                for p, p_targ in zip(q.parameters(), q_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
            # TODO: TEST: for p in q_targ.parameters():
                # print(p.requires_grad)

    def get_action(o, deterministic=False, epsilon=0.0):
        return q.act(torch.as_tensor(o, dtype=torch.float32), deterministic, epsilon)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take greedy actions at test time
                o, r, d, _ = test_env.step(get_action(o, deterministic=eval_greedy, epsilon=epsilon))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)



    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    # Define a schedule for (decreasing) exploration
    exploration = core.LinearSchedule(schedule_timesteps=int(exploration_fraction * total_steps),
                                      initial_p=exploration_initial_eps,
                                      final_p=exploration_final_eps)

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        epsilon = exploration.value(t)

        if t > start_steps:
            a = get_action(o, deterministic=False, epsilon=epsilon)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # print(info)
        # d = False if ep_len == max_ep_len else d
        d = False if info.get('TimeLimit.truncated', False) else d
        # print(f"d = {d}; ep_len = {ep_len}")

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            if d and r == 100:
                print(f"True end of episode and reward was {r}!!")
            # print("-----------------------------------------------")
            # print("------------------RESTART----------------------")
            # print(f"d = {d}; ep_len = {ep_len}")
            # print("-----------------------------------------------")
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            print(f"Epsilon is currently {epsilon}.")

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                save_time = time.time()
                logger.save_state({'env': env}, None)
                # print(f"Saving took {(time.time()-save_time)} seconds.")
            # Test the performance of the greedy version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('TestEpRet')
            logger.log_tabular('EpRet')
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', (time.time()-start_time)/60)
            logger.log_tabular('Epoch', epoch)
            logger.dump_tabular()


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=256)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='dqn')
#     args = parser.parse_args()
#
#     from spinup.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
#
#     torch.set_num_threads(torch.get_num_threads())
#
#     dqn(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
#         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#         logger_kwargs=logger_kwargs)
