import pandas as pd
import numpy as np
import os

from typing import NamedTuple, Tuple, Dict, List, Union
from functools import partial

from gym_PGFS.datasets import get_fingerprint_fn

from numpy.random import RandomState
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from gym_PGFS.scorers.parametersearch import ParameterSearch, define_search_grid

# logging
from aim import Run


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
                 loss_fns: Dict = {}):
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
    loss_fns
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
            loss_fns
        )

        # save the results to the data
        result_list.append(fold_outcome)

    return result_list



def train_one_fold(X: np.ndarray, y: np.ndarray,
                   X_valid: np.ndarray, y_valid: np.ndarray,
                   hyperparams: Dict,
                   loss_fns: Dict = {}):
    '''
    Trains and evaluates one fold.
    Parameters
    ----------
    X
    y
    X_valid
    y_valid
    loss_fns
    hyperparams

    Returns
    -------
        A tuple containing dictionaries: (train_losses, validation_losses)
    '''
    clf = RandomForestClassifier(**hyperparams)
    clf.fit(X, y)

    # compute the train and validation loss
    train_losses = {"score.train_"+k: loss_fn(y, clf.predict(X)) for k, loss_fn in loss_fns.items()}
    validation_losses = {"score.valid_"+k: loss_fn(y_valid, clf.predict(X_valid)) for k, loss_fn in loss_fns.items()}

    # return a combined fold entry that will go into a dataframe
    retdict = {}
    retdict.update(train_losses)
    retdict.update(validation_losses)
    return retdict


def hyperparameter_eval(X, y,
                        splits_seed: int,
                        hparams_source: ParameterSearch,
                        n_folds = 9,
                        experiment_name = 'opt',
                        aim_record_dir = './',
                        extras: Dict = None
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
    X_train, y_train, X_test, y_test = minor_split(X, y, test_size=0.1, rs=splits_seed)

    # enter the loop
    id, hp = hparams_source.get_next_setting()
    while id is not None and hp is not None:
        # evaluate the hyperparameter set
        losses = train_one_cv(X_train,
                              y_train,
                              hp, n_folds=n_folds,
                              random_seed_folds=splits_seed,
                              loss_fns={
                                  'roc_auc': roc_auc_score,
                                  'bal_acc': balanced_accuracy_score,
                                  'accuracy': accuracy_score
                              })

        # return the results to the hyperparameter server
        hparams_source.submit_result(id, losses)

        # record the findings in aim
        run = Run(experiment=experiment_name,repo=aim_record_dir)
        run['hparams'] = hp
        if extras:
            for key,val in extras.items():
                run[key] = val
        for i, loss in enumerate(losses):  # the order of cv folds is deterministic with fixed seed
            for key, val in loss.items():
                run.track(val, name=key, step=i, context={'subset': 'train'})
        run.close()

        # try to get the next hyperparams
        id, hp = hparams_source.get_next_setting()

    return True


def train_rfc_mgenfail(is_server: bool,
                       hosts: Dict,
                       prefix: str = './data/mgenfail_essays',
                       dataset: str = 'CHEMBL1909140',
                       descriptors: str = 'ECFP_2_1024',
                       n_folds: int = 9,
                       ):
    if is_server:
        hparam_ranges = {
            'n_estimators': [5, 10, 15, 20, 35, 50, 60, 75, 90, 100, 150, 200,],
            'random_state': [0xDEADBEEF, 0xBADACC, 228],
        }
        # initialize the first parameter server
        hp_server = define_search_grid(hparam_ranges, prefix+'/OS_SCORE')
        # if server, then hosts is just an int with a port
        # start the server in the same thread, so that we know, when it runs out of parameters
        hp_server.start_server(hosts['server'], hosts['port'], as_thread=False)
    else:
        df = load_processed_dataset(dataset, prefix)
        df['descriptors'] = compute_descriptors(df.smiles, descriptors)

        X_mod, y_mod, X_data, y_data = prepare_major_splits(df)

        # connect ot the hyperparameter server
        hp_source = ParameterSearch(host=hosts['server'], port=hosts['port'])

        # train the OS model
        hyperparameter_eval(X_mod,
                            y_mod,
                            228,
                            hp_source,
                            n_folds = n_folds,
                            experiment_name='OS_MODEL',
                            aim_record_dir=prefix,
                            extras={'dataset': {'name': dataset, 'descriptors': descriptors}}
                            )
        # the MCS model is a model with the same parameters as the OS model, but seeded differently

        # TODO: train the DS model