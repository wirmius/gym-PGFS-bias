import os

import numpy as np

from typing import Tuple, Union, Dict

from matplotlib import pyplot as plt

from ..config import load_config
from ..envs.PGFS_trials import PGFS_env, gym_PGFS_basic_from_config
from ..rl.agents import PGFS_agent


def record_rewards_from_env(env: PGFS_env, agent: PGFS_agent = None, n_samples=100):
    retarray = np.zeros((env.max_steps+1, n_samples))
    ep_counter = 0
    while ep_counter < n_samples:
        s = env.reset()
        retarray[0, ep_counter] = env.scoring_fn(env.rmodel.current_mol)
        try:
            for i in range(1, env.max_steps+1):
                if agent:
                    a = agent.act(*s)
                else:
                    a = env.suggest_action()

                s, r, done, _ = env.step(a)
                retarray[i, ep_counter] = r
        except:
            continue

        ep_counter += 1
    # return the rewards
    return retarray.T


def compare_against_random(env: Union[PGFS_env, str, os.PathLike],
                           agent: Union[PGFS_agent, str, os.PathLike],
                           run_config: Dict = None,
                           n_samples=100):
    if not env or isinstance(env, str) or isinstance(env, os.PathLike):
        cfg = load_config(env)
        env = gym_PGFS_basic_from_config('data', cfg)
    if not run_config:
        run_config = load_config()['run']
    if isinstance(agent, str) or issubclass(agent, os.PathLike):
        agent_fname = agent
        agent = PGFS_agent(env.action_space,
                           env.observation_space,
                           env.rng,
                           **run_config,
                           gumbel_tau=run_config['g_tau_start'])
        # not forgetting to load the checkpoint specified
        agent.load_checkpoint(agent_fname)

    # run out the random search and keep the results in the ndarray
    random_scores = record_rewards_from_env(env, None, n_samples)

    # run out the agent and keep the results in an ndarray
    agent_scores = record_rewards_from_env(env, agent, n_samples)

    return agent_scores, random_scores


def draw_comparison_figure(random_sample: np.ndarray,
                           agent_sample: np.ndarray,
                           size: Tuple = (10, 7),
                           ylabel = ""):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot()
    plt.ylabel(f"{ylabel} \n(n_samples = {random_sample.shape[0]})")
    plt.xlabel("step #")
    c, c_highlight = 'gray', 'black'
    rng = np.arange(random_sample.shape[1])
    random_box = ax.boxplot(random_sample, patch_artist=True, positions=rng-0.15, widths=0.2,
                             boxprops=dict(facecolor=c, color=c),
                             capprops=dict(color=c_highlight),
                             whiskerprops=dict(color=c_highlight),
                             flierprops=dict(color=c, markeredgecolor=c),
                             medianprops=dict(color=c_highlight))
    c, c_highlight = 'orange', 'red'
    agent_box = ax.boxplot(agent_sample, patch_artist=True, positions=rng+0.15, widths=0.2,
                             boxprops=dict(facecolor=c, color=c),
                             capprops=dict(color=c_highlight),
                             whiskerprops=dict(color=c_highlight),
                             flierprops=dict(color=c, markeredgecolor=c),
                             medianprops=dict(color=c_highlight))
    ax.legend([random_box["boxes"][0], agent_box["boxes"][0]],
              ['random sample', 'agent'],
              loc='upper left')

    ax.xaxis.set_ticks(rng)
    ax.set_xticklabels(rng)
    #ax.legend()



    return fig
