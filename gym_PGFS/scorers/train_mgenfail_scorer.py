import pickle
import time

import pandas as pd
import numpy as np
import os

from typing import NamedTuple, Tuple, Dict, List, Union
from functools import partial

from gym_PGFS.datasets import get_fingerprint_fn
from gym_PGFS.utils import ensure_dir_exists

from numpy.random import RandomState
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from gym_PGFS.scorers.parametersearch import ParameterSearch, define_search_grid
import pickle

# logging
from aim import Run

# hyperopt
from hyperopt import hp
from hyperopt.base import STATUS_OK
from collections import defaultdict


def load_processed_dataset(chid: str, datadir: str) -> pd.DataFrame:
    assay_file = os.path.join(datadir, f'processed/{chid}.csv')
    return pd.read_csv(assay_file)


def compute_descriptors(smiles: pd.Series, fp_type: str) -> pd.Series:
    fn, params = get_fingerprint_fn(fp_type)
    fn = partial(fn, **params)
    return smiles.apply(fn, convert_dtype=np.ndarray)


def get_dataset_as_numpy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    '''

    Parameters
    ----------
    df

    Returns
    -------
        X and y matrices
    '''
    return np.stack(df.descriptors.to_list()), df.label.to_numpy()


def prepare_major_splits(df: pd.DataFrame, rs: Union[RandomState, int] = 0) -> Dict:
    df1, df2 = train_test_split(df, test_size=0.5, stratify=df['label'], random_state=rs)

    X_models, y_models = get_dataset_as_numpy(df1)
    X_data, y_data = get_dataset_as_numpy(df2)

    return X_models, y_models, X_data, y_data


def minor_split(X, y, test_size = 0.1, rs: Union[RandomState, int] = 228):
    # initialize the random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)
    return X_train, y_train, X_test, y_test


def train_one_cv(X_train, y_train,
                 hyperparams: Dict,
                 n_folds: int = 9,
                 random_seed_folds: Union[RandomState, int] = 0,
                 score_fns: Dict = {}):
    '''
    Evaluates one hyperparameter set. RandomState for the model should be integrated into the hparams

    Parameters
    ----------
    X_train
    y_train
    hyperparams
        A dictionary of hyperparemeters to evaluate
    n_folds
        number of folds
    random_seed_folds
        the seed used by the stratified K fold generator
    score_fns
        loss functions to evaluate

    Returns
    -------
str
    '''


    # get stratified k folds
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed_folds)
    result_list = []
    for fold_count, (train_index, val_index) in enumerate(folds.split(X_train, y_train)):

        fold_outcome = train_one_fold(
            X_train[train_index],
            y_train[train_index],
            X_train[val_index],
            y_train[val_index],
            hyperparams,
            score_fns
        )

        # save the results to the data
        train_loss, val_loss = fold_outcome
        result_list.append({'train': train_loss, 'validation': val_loss})

    return result_list



def train_one_fold(X: np.ndarray, y: np.ndarray,
                   X_valid: np.ndarray, y_valid: np.ndarray,
                   hyperparams: Dict,
                   score_fns: Dict = {}):
    '''
    Trains and evaluates one fold.
    Parameters
    ----------
    X
    y
    X_valid
    y_valid
    score_fns
    hyperparams

    Returns
    -------
        A tuple containing dictionaries: (train_losses, validation_losses)
    '''
    clf = RandomForestClassifier(**hyperparams)
    clf.fit(X, y)

    # compute the train and validation loss
    train_losses = {k: score_fn(y, clf.predict(X)) for k, score_fn in score_fns.items()}
    validation_losses = {k: score_fn(y_valid, clf.predict(X_valid)) for k, score_fn in score_fns.items()}

    # return a combined fold entry that will go into a dataframe
    return train_losses, validation_losses


def hyperparameter_eval(X, y,
                        splits_seed: int,
                        hparams_source: ParameterSearch,
                        n_folds = 9,
                        experiment_name = 'opt',
                        aim_record_dir = './',
                        extras: Dict = None,
                        score_fns: dict = {'roc_auc': roc_auc_score},
                        score_to_select: str = 'roc_auc'
                        ):
    '''

    Parameters
    ----------
    X - data to run the cross validation on
    y - labels
    splits_seed
    hparams_source
    n_folds
    experiment_name
    aim_record_dir
    extras - a dictionary of dictionaries of other things to include in the logging with aim

    Returns
    -------

    '''

    # generate the split of the train/test set
    X_train, y_train, = X, y

    # enter the loop
    id, hp = hparams_source.get_next_setting()

    while id is not None and hp is not None:
        # quick and dirty fix to fix an error of floats being fed into integer params
        hp = {key: int(val) for key, val in hp.items()}
        # evaluate the hyperparameter set

        scores = train_one_cv(X_train,
                              y_train,
                              hp, n_folds=n_folds,
                              random_seed_folds=splits_seed,
                              score_fns = score_fns)

        # record the findings in aim
        run = Run(experiment=experiment_name, repo=aim_record_dir)
        run['hparams'] = hp
        if extras:
            for key, val in extras.items():
                run[key] = val

        track = defaultdict(int)
        for key in score_fns.keys():
            for i, score in enumerate(scores):  # the order of cv folds is deterministic with fixed seed
                train_score = score['train']
                val_score = score['validation']
                track['train_mean_' + key] += train_score[key] / len(scores)
                track['valid_mean_' + key] += val_score[key] / len(scores)

                run.track(train_score[key], key, step=i, context={'metric_over': 'train'})
                run.track(val_score[key], key, step=i, context={'metric_over': 'valid'})
        for key, val in track.items():
            run.track(val, key, context={'metric_over': 'mean'})
        run.close()

        # return the results to the hyperparameter server
        hparams_source.submit_result(id, {'loss': 1-track['valid_mean_'+score_to_select], 'status': STATUS_OK})

        # try to get the next hyperparams
        id, hp = hparams_source.get_next_setting()

    return True


def train_final_and_test(X_train, y_train,
                         X_test, y_test,
                         hparams,
                         score_fns: dict = {'roc': roc_auc_score} ):

    # quick and dirty fix to fix an error of floats being fed into integer params
    hparams = {key: int(val) for key, val in hparams.items()}

    clf = RandomForestClassifier(**hparams)
    clf.fit(X_train, y_train)
    ret_loss = {name: score(y_test, clf.predict(X_test)) for name, score in score_fns}
    return ret_loss, clf


def train_final_test_and_dump(X_train, y_train,
                              X_test, y_test,
                              prefix,
                              best_params,
                              score_fns: dict = {'roc': roc_auc_score} ):
    score, clf = train_final_and_test(X_train, y_train,
                                      X_test, y_test,
                                      best_params,
                                      score_fns=score_fns)
    # log and print the best score
    print(f"Best parameter set evaluated: {best_params}")
    print(f"Resulting model has test scores of:\n {score}.")
    # pickle the model and record the best parameter set
    with open(prefix + '/model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(prefix + '/best_parameters.txt', 'wt') as f:
        f.writelines([str(best_params), str(score)])
    return score, clf



def train_rfc_mgenfail(is_server: bool,
                       hosts: Dict,
                       prefix: str = './data/mgenfail_essays',
                       dataset: str = 'CHEMBL1909140',
                       descriptors: str = 'ECFP_2_1024',
                       n_folds: int = 6,
                       n_tries: int = 1000,
                       worker_break = 60.,
                       ):
    df = load_processed_dataset(dataset, prefix)
    df['descriptors'] = compute_descriptors(df.smiles, descriptors)

    X_mod, y_mod, X_data, y_data = prepare_major_splits(df)
    #
    # print(X_mod.shape, y_mod.shape)
    # print(X_mod.sum(), y_mod.sum())
    # print(X_data.sum(), y_data.sum())

    score_fns = {
        'roc_auc': roc_auc_score,
        'bal_acc': balanced_accuracy_score,
        'accuracy': accuracy_score
    }
    select_by_score = 'roc_auc'

    # now change the prefix to be more specific
    prefix += '/' + dataset

    if is_server:
        hparams = {
            'n_estimators': hp.quniform('n_estimators', 5, 100, 1),
            'max_features': hp.quniform('max_features', 10, 30, 1),
            'random_state': 0xDEADBEEF
        }
        ensure_dir_exists(prefix)
        ensure_dir_exists(prefix + '/OS_MODEL')
        ensure_dir_exists(prefix + '/DCS_MODEL')
        ensure_dir_exists(prefix + '/MCS_MODEL')
        # initialize the first parameter server
        hp_server = ParameterSearch(space = hparams, rng = RandomState(553),
                                    tries = n_tries, output_file = prefix+'/OS_MODEL/hyperparameter_log.txt')
        # if server, then hosts is just an int with a port
        # start the server in the same thread, so that we know, when it runs out of parameters
        hp_server.start_server(hosts['server'], hosts['port'], as_thread=False)

        # get the best hyperparams, train the model, dump it where it should be
        os_best_params = hp_server.get_results()
        train_final_test_and_dump(X_mod, y_mod,
                                  X_data, y_data,
                                  prefix + '/OS_MODEL',
                                  os_best_params,
                                  score_fns = score_fns
                                  )

        # now train and dump the MCS model
        mcs_best_params = hp_server.get_results()
        # initialise the model randomly with a different seed
        mcs_best_params['random_state'] = np.random.randint(4096)
        train_final_test_and_dump(X_mod, y_mod,
                                  X_data, y_data,
                                  prefix + '/MCS_MODEL',
                                  mcs_best_params,
                                  score_fns = score_fns
                                  )
        # now proceed to train the DCS model
        # initialize the second parameter server
        hp_server = ParameterSearch(hparams, RandomState(553), n_tries, output_file=prefix+'/DCS_MODEL/hyperparameter_log.txt')
        # if server, then hosts is just an int with a port
        # start the server in the same thread, so that we know, when it runs out of parameters
        hp_server.start_server(hosts['server'], hosts['port'], as_thread=False)
        # now train and dump the DCS model
        dcs_best_params = hp_server.get_results()
        train_final_test_and_dump(X_data, y_data,
                                  X_mod, y_mod,
                                  prefix + '/DCS_MODEL',
                                  dcs_best_params,
                                  score_fns = score_fns
                                  )

        # done
    else:
        # connect ot the hyperparameter server
        hp_source = ParameterSearch(host=hosts['server'], port=hosts['port'])

        # train the OS model
        hyperparameter_eval(X_mod,
                            y_mod,
                            splits_seed=228,
                            hparams_source=hp_source,
                            n_folds = n_folds,
                            experiment_name='OS_MODEL',
                            aim_record_dir=prefix,
                            score_fns=score_fns,
                            score_to_select=select_by_score,
                            extras={'dataset': {'name': dataset, 'descriptors': descriptors}}
                            )
        # skip a minute
        time.sleep(worker_break)

        # go on to optimise the Data Control Model
        hyperparameter_eval(X_data,
                            y_data,
                            splits_seed=228,
                            hparams_source=hp_source,
                            n_folds = n_folds,
                            experiment_name='DCS_MODEL',
                            aim_record_dir=prefix,
                            score_fns=score_fns,
                            score_to_select=select_by_score,
                            extras={'dataset': {'name': dataset, 'descriptors': descriptors}}
                            )

if __name__ == '__main__':
    from sys import argv

    if not argv[1] in ['server', 'client']:
        raise ValueError("must specify the role correctly: either server or client")
    is_server = argv[1] == 'server'
    assay = argv[2]
    n_folds = int(argv[3])
    n_tries = int(argv[4])
    host = argv[5]
    port = int(argv[6])
    # example usage (in the root of the package):
    # python gym_PGFS/scorers/train_mgenfail_scorer.py server CHEMBL1909140 6 3000 127.0.0.1 5533
    # python gym_PGFS/scorers/train_mgenfail_scorer.py client CHEMBL1909140 6 3000 127.0.0.1 5533

    train_rfc_mgenfail(is_server,
                       dataset=assay,
                       hosts={'server': '127.0.0.1', 'port': 5555},
                       n_tries=n_tries,
                       n_folds=n_folds
                       )