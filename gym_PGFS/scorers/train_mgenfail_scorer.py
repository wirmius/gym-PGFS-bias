from hyperopt import hp
import pandas as pd
import numpy as np
import os

from typing import NamedTuple, Tuple, Dict, List, Union
from functools import partial

from gym_PGFS.datasets import get_fingerprint_fn

from numpy.random import RandomState
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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


def prepare_major_splits(df: pd.DataFrame, rs: Union[RandomState, str, int] = "") -> Dict:
    # initialize the random state
    if not isinstance(rs, RandomState):
        rs = RandomState(seed=rs)

    df1, df2 = train_test_split(df, test_size=0.5, stratify=df['label'])

    X_models, y_models = get_dataset_as_numpy(df1)
    X_data, y_data = get_dataset_as_numpy(df2)

    return X_models, y_models, X_data, y_data


def minor_split(X, y, test_size = 0.9, rs: Union[RandomState, str, int] = ""):
    # initialize the random state
    if not isinstance(rs, RandomState):
        rs = RandomState(seed=rs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)
    return X_train, y_train, X_test, y_test


def train_one_cv(X_train, y_train,
                 hyperparams: Dict,
                 recording: pd.DataFrame,
                 n_folds: int = 9,
                 random_state: Union[RandomState, str, int] = "",
                 random_seed_models: Union[RandomState, str, int] = "",
                 loss_fns: Dict = {}):

    # initialize the random state
    if not isinstance(random_state, RandomState):
        random_state = RandomState(seed=random_state)

    # get stratified k folds
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_index, val_index in folds.split(X_train, y_train):

        # supplying a random_state object enables state that evolves between the runs
        model_state = RandomState(seed=random_state) if not isinstance(random_seed_models, RandomState) \
            else random_seed_models
        hyperparams['random_state'] = model_state

        train_one_fold(X_train[train_index], y_train[train_index],
                       X_train[val_index], y_train[val_index],
                       hyperparams,
                       loss_fns)

        # TODO: address having random state in the dictionary

        # save the results to the data


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
    train_losses = {"train_"+k: loss_fn(y, clf.predict(X)) for k, loss_fn in loss_fns.items()}
    validation_losses = {"valid_"+k: loss_fn(y_valid, clf.predict(X_valid)) for k, loss_fn in loss_fns.items()}

    return train_losses, validation_losses


def hyperparameter_search(X, y,
                          run: Run,
                          hparam_grid: Dict):
    pass


def train_rfc_mgenfail(prefix: str = './data/mgenfail_essays',
                       dataset: str = 'CHEMBL1909140',
                       n_folds: int = 9,
                       ):
    df = load_processed_dataset(dataset, prefix)
    df['descriptors'] = compute_descriptors(df.smiles, "ECFP_2_1024")
    X, y = get_dataset_as_numpy(df)

    X_mod, y_mod, X_data, y_data = prepare_major_splits(X, y)



    # TODO: train the OS model

    # TODO: train the MCS model

    # TODO: train the DS model