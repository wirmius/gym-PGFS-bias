import pandas as pd
import numpy as np

from typing import Tuple, List, Dict

from matplotlib import pyplot as plt
from tqdm import tqdm

from gym_PGFS.scorers.scorer import ScorerModeSelector, GuacamolMGenFailScorer


# collect data from a single
def process_output_into_dataframe(list_o_lists: List[List[str]], scorer: GuacamolMGenFailScorer):
    modes = [ScorerModeSelector.OS_MODE, ScorerModeSelector.MCS_MODE, ScorerModeSelector.DCS_MODE]

    ret_df = pd.DataFrame({})

    for epoch, smi_list in tqdm(enumerate(list_o_lists)):
        for mode in modes:
            # evaluate the list of smiles
            scores_this_epoch = {}

            scores_this_epoch['smiles'] = smi_list
            scores_this_epoch['epoch'] = [epoch for smi in smi_list]
            scores_this_epoch['type'] = [str(mode)[19:] for smi in smi_list]

            scorer.mode = mode
            scores_this_epoch['value'] = scorer.raw_score_list(smi_list)

            ep_df = pd.DataFrame.from_dict(scores_this_epoch, orient='index').transpose()
            ret_df = pd.concat([ret_df, ep_df])

    return ret_df.reset_index().drop('index', 1)


# similar routine but for batch processing smiles from the PGFS
def process_PGFS_output_into_dataset(PGFS_output: Dict, scorer: GuacamolMGenFailScorer):
    modes = [ScorerModeSelector.OS_MODE, ScorerModeSelector.MCS_MODE, ScorerModeSelector.DCS_MODE]
    accumulator = pd.DataFrame()

    for iter, list_o_lists in PGFS_output.items():
        agent_state_accumulator = pd.DataFrame()
        for episode_outcome in list_o_lists:
            scores_this_episode = {}

            scores_this_episode['smiles'] = episode_outcome
            scores_this_episode['epoch'] = [iter for smi in episode_outcome]
            scores_this_episode['score'] = [str(mode)[19:] for smi in episode_outcome]
            # the largest difference from the lstm output: parse also all of the stages
            scores_this_episode['step'] = range(len(episode_outcome))

            for mode in modes:
                scorer.mode = mode
                scores_this_episode['value'] = scorer.raw_score_list(scores_this_episode)

            ep_df = pd.DataFrame.from_dict(scores_this_episode, orient='index').transpose()
            agent_state_accumulator = pd.concat([agent_state_accumulator, ep_df])

        accumulator = pd.concat([accumulator, agent_state_accumulator])

    return accumulator.reset_index().drop('index', 1)


def plot_medians(df: pd.DataFrame,
                 ax,
                 position: Tuple[int, int],
                 dataset_name: str,
                 y_label: str = 'Value',
                 x_label: str = 'Step',
                 y_scale: Tuple[float, float] = None,
                 x_scale: Tuple[float, float] = None):
    # introduce mode, q1, q3 inside each epoch and type
    epochs = df.epoch.unique()
    types = df.type.unique()
    dct = {'epoch' :[] ,'type' :[] ,'q1' :[] ,'median' :[] ,'q3' :[]}
    for t in types:
        typed = df[df['type' ]==t]
        for e in epochs:
            s = typed[typed['epoch' ]==e]['value']
            q1 = s.quantile(.25)
            q2 = s.quantile(.5)
            q3 = s.quantile(.75)
            dct["type"].append(t)
            dct["epoch"].append(e)
            dct["q1"].append(q1)
            dct["q3"].append(q3)
            dct["median"].append(q2)
    # create aggregated df
    aggr_df = pd.DataFrame(dct)
    aggr_df.head()

    # plot
    cs = ['indigo' ,'yellowgreen' ,'hotpink']
    labels = ["OS" ,"MCS" ,"DCS"]
    dashes = ["-" ,"--" ,"-."]
    for i ,t in enumerate(types):
        sub = aggr_df[aggr_df['type' ]==t]
        x = sub['epoch']
        y = sub['median']
        y_q1 = sub['q1']
        y_q3 = sub['q3']
        ax.plot(x, y, dashes[i], c=cs[i], label=labels[i])  # plot median
        ax.fill_between(x, y_q1, y_q3, alpha=0.1, color=cs[i])  # plot iqr

    # set the axis scales
    if x_scale:
        ax.set_xlim(x_scale)
    if y_scale:
        ax.set_ylim(y_scale)

    # add the axis labels
    ax.set_xlabel(x_label)
    if y_label and not position[0]:
        ax.set_ylabel(y_label)

    # set title
    ax.set_title(dataset_name)

    # generate the legend
    ax.legend()

    # return something ( not sure about it yet )
    return ax
