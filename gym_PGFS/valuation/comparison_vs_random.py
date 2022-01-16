import os

import numpy as np

from typing import Tuple, Union, Dict

from matplotlib import pyplot as plt

from ..config import load_config
from ..envs.PGFS_trials import PGFS_env, gym_PGFS_basic_from_config
from ..rl.agents import PGFS_agent

from tqdm import tqdm


def record_rewards_from_env(env: PGFS_env, agent: PGFS_agent = None, n_samples=100):
    retarray = np.zeros((env.max_steps+1, n_samples))
    ep_counter = 0
    while ep_counter < n_samples:
        s = env.reset()
        retarray[0, ep_counter] = env.scoring_fn.present_score(env.scoring_fn.score(env.rmodel.current_mol))
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


from gym_PGFS.scorers.scorer import ScorerModeSelector, MGenFailScorer
def compare_mgenfail_mode(env: Union[PGFS_env, str, os.PathLike],
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

    assert isinstance(env.scoring_fn, MGenFailScorer)

    # run the agent with the os score
    env.scoring_fn.set_mode(ScorerModeSelector.OS_MODE)
    os_scores = record_rewards_from_env(env, agent, n_samples)

    # evaluate on the mcs score
    env.scoring_fn.set_mode(ScorerModeSelector.MCS_MODE)
    mcs_scores = record_rewards_from_env(env, agent, n_samples)

    # evaluate on the dcs score
    env.scoring_fn.set_mode(ScorerModeSelector.DCS_MODE)
    dcs_scores = record_rewards_from_env(env, agent, n_samples)

    env.scoring_fn.set_mode(ScorerModeSelector.OS_MODE)
    return os_scores, mcs_scores, dcs_scores


def compare_over_timesteps(env: Union[PGFS_env, str, os.PathLike],
                           agents: Union[os.PathLike],
                           run_config: Dict = None,
                           n_samples=100):
    assert os.path.exists(agents) and os.path.isdir(agents)

    # load agent states one by one and evaluate
    agent_filenames = [fname for fname in os.listdir(agents) if os.path.isfile(fname)]

    return_dict = {}

    for afile in agent_filenames:
        agent_full_path = os.path.join(agents, afile)
        agent = PGFS_agent(env.action_space,
                           env.observation_space,
                           env.rng,
                           **run_config,
                           gumbel_tau=run_config['g_tau_start'])
        # load the checkpoint
        agent.load_checkpoint(agent_full_path)

        # get the point of evaluation
        agent_episode = int(afile.split('.')[0].split('_')[1])

        # conduct an experiment
        os_score, mcs_score, dcs_score = compare_mgenfail_mode(env, agent, run_config, n_samples)

        return_dict[agent_episode] = {'os': os_score, 'mcs': mcs_score, 'dcs': dcs_score}

    return return_dict


def draw_mgen_comparison_figure(os_sample: np.ndarray,
                                mcs_sample: np.ndarray,
                                dcs_sample: np.ndarray,
                                size: Tuple = (10, 7),
                                ylabel = "",
                                xlabel = ""):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot()
    plt.ylabel(f"{ylabel} \n(n_samples = {os_sample.shape[0]})")
    plt.xlabel(xlabel)
    c, c_highlight = 'cornflowerblue', 'dimgrey'
    rng = np.arange(os_sample.shape[1])
    os_box = ax.boxplot(os_sample, patch_artist=True, positions=rng-0.15, widths=0.2,
                        boxprops=dict(facecolor=c, color=c),
                        capprops=dict(color=c_highlight),
                        whiskerprops=dict(color=c_highlight),
                        flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color=c_highlight))
    c, c_highlight = 'navy', 'darkblue'
    mcs_box = ax.boxplot(mcs_sample, patch_artist=True, positions=rng+0.15, widths=0.2,
                         boxprops=dict(facecolor=c, color=c),
                         capprops=dict(color=c_highlight),
                         whiskerprops=dict(color=c_highlight),
                         flierprops=dict(color=c, markeredgecolor=c),
                         medianprops=dict(color=c_highlight))
    c, c_highlight = 'red', 'orange'
    dcs_box = ax.boxplot(dcs_sample, patch_artist=True, positions=rng+0.15, widths=0.2,
                         boxprops=dict(facecolor=c, color=c),
                         capprops=dict(color=c_highlight),
                         whiskerprops=dict(color=c_highlight),
                         flierprops=dict(color=c, markeredgecolor=c),
                         medianprops=dict(color=c_highlight))
    ax.legend([os_box["boxes"][0], mcs_box["boxes"][0], dcs_box["boxes"][0]],
              ['optimization score', 'model control score', 'data control score'],
              loc='upper left')

    ax.xaxis.set_ticks(rng)
    ax.set_xticklabels(rng)

    return fig