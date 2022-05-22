# Satisficing policies for Markov Decision Processes

This repository is the official implementation of "Satisficing policies for Markov Decision Processes" (under review).

This repo contains:

- Code (and commands, see below) to reproduce the empirical results of Figures 2
  and 3 in Section 7 of the article.
- An implementation of the Macro-Action Gridworld described in Section 7 and
  used in Figure 2. The environment is using the
  [OpenAI-Gym](https://github.com/openai/gym) API and thus can be easily
  used in other experiments and for other algorithms.
- A pre-trained double-DQN model for the LunarLander environment that was used
  as approximate optimal value function for all policies tested in Figure 3.

## Requirements / installation

The easiest way to use this repository is to create a `conda` environment (here the environment is
called "satisficing") via:

```setup
conda create -n satisficing python=3.6
```

Then activate the environment using:

```setup
conda activate satisficing
```

Clone this repo and then `cd` into it

```setup
git clone https://github.com/satisficing-mdp/satisficing.git
cd satisficing
```

where you now can easily install all requirements via

```setup
pip install -e .
```

## Reproducing results

### Macro-action Gridworld (Figure 2)

Figure 2 compares different policies for a given optimal, tabular Q value
function, which was computed via value iteration. Use the following scripts to
evaluate value-tracking satisficing...

```
python spinup/run.py test_xis_for_q_star_pytorch --env FourActionWorld-v0 --exp_name q_star --seed 0to1 --add_date --eval_greedy False --num_eval_runs 1000 --max_ep_len 50 --xi 0 --xi_update_fn decrease_by_reward
```

... and $\epsilon$-greedy policies for $\epsilon = 0.0, 0.1, 0.2, \dots, 1.0$,
that is including the greedy ($\epsilon = 0.0$) and random ($\epsilon = 1.0$) policies...

```
python spinup/run.py test_xis_for_q_star_pytorch --env FourActionWorld-v0 --exp_name q_star_eps_gre --seed 0to1 --add_date --eval_greedy False --plot_tradeoffs --num_eval_runs 1000 --max_ep_len 50 --actor core.TabSARSA --epsilon 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

### Lunar-Lander (Figure 3)

Figure 3 compares different policies for a given approximately optimal Q value
function. To reproduce the results, you can use a pretrained model
artefact (learned parameters of a neral network learned via double DQN, trained
using OpenAI's [spinningup](https://github.com/openai/spinningup/)
implementation), which is stored in the `pretrained_models` directory.

To compute the results for value-tracking and valved value-tracking $\xi$-satisficing policies use

```setup
python spinup/run.py exploit_pytorch --exp_name ll_eps_satisficing --fpath pretrained_models/LunarLander-DDQN/policy --env_name LunarLander-v2 --add_date --epsilon_greedy False --epsilon 0.0 --start_xi_fn core.use_max_q_value --xi_update_fn core.value_tracking core.no_negative_value_tracking
```

To compute results for the baseline greedy and random policies use

```setup
python spinup/run.py exploit_pytorch --exp_name ll_eps_greedy --fpath pretrained_models/LunarLander-DDQN/policy --env_name LunarLander-v2 --add_date --epsilon_greedy True --epsilon 0.0 1.0
```

The results are automatically stored into a newly created `data/` directory within this repository.

Alternatively, you can first retrain (warning, this takes some time!) the
approximate Q-value function using

```setup
python spinup/run.py dqn_pytorch --exp_name dqn_ll --q_net core.MLPQFunction --env LunarLander-v2 --steps_per_epoch 10000 --epochs 60 --plot_lc --start_steps 100 --seed 0to1 --add_date --vf_lr 0.001 --update_every 1 --update_target_network_every 100 --save_freq 10 --eval_greedy True --num_test_episodes 20 --max_ep_len 1000 --double_q_learning --q_kwargs:hidden_sizes "[64]" --exploration_final_eps 0.05
```

which creates a model artefact in the `data/` folder, which then can be targeted
by the `exploit` commands shown above.

### Reading results

All run commands shown above create one folder for each policy evaluated in the
`data/` folder with info about the evaluation run as well as a timestamp. Each
of these folders contain results in a `progress.txt` file. Specifically, each
file contains information about

- **quality** of the policy in terms of mean reward obtained
  (`AverageTestEpRet`), and corresponding standard deviation (`StdTestEpRet`), as
  well as
- **effort** required by the policy in terms of average number of actions
  considered (`AverageTestEpActions`) and corresponding standard deviation (`StdTestEpActions`).

## Credits and references

- All code related to $\xi$-satisficing policies was implemeted by us.
- The MacroAction Gridworld was implemented by us.
- We used [OpenAI Gym](https://github.com/openai/gym) for the LunarLander domain
- We used [OpenAI SpinningUp](https://github.com/openai/spinningup/) for the double DQN implementation.

See the `LICENSE` file for information about the MIT License under which this
repository is published.
