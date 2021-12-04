from gym import Space
from gym.spaces.multi_binary import MultiBinary
import numpy as np
import pandas as pd
from typing import List, Callable, Union

import os
from datetime import datetime



def readable_time_now() -> str:
    '''
    Returns
    -------
        current time in a readable format
    '''
    return datetime.now().strftime('__%m_%d_%H_%M')


def shelf_directory(dir_path: Union[os.PathLike, str]) -> str:
    '''
    Parameters
    ----------
    dir_path: path/string
        the directory to be renamed
    Returns
    -------
        the new name with readable timestamp that the directory was renamed to
    '''
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        new_path = str(dir_path)+readable_time_now()
        os.rename(dir_path, new_path)
    else:
        new_path = None
    os.mkdir(dir_path)

    return new_path


def ensure_dir_exists(dir_path: Union[os.PathLike, str]) -> bool:
    '''
    Parameters
    ----------
    dir_path
        directory path
    Returns
    -------
        True if the directory already existed before
    '''
    if os.path.exists(dir_path):
        return True
    else:
        os.mkdir(dir_path)
        return False



def np_to_onehot(targets: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[targets]


def np_from_onehot(ohenc: np.ndarray) -> np.ndarray:
    # possible bug here
    return np.nonzero(ohenc)[-1]


class MolVector(MultiBinary):
    def __init__(self, molVecSize):
        '''Is a gym Space'''
        super().__init__(molVecSize)


class ReactionFailed(Exception):
    def __init__(self, rxn_smarts: str, reactants: List[str], rxn_id: int, others: str = ""):
        super(ReactionFailed, self).__init__()
        self.rxn = rxn_smarts
        self.reactants = reactants
        self.others = others
        self.rxn_id = rxn_id

    def __str__(self):
        return super().__str__()+"Template ID: {} \n Reactions SMARTS: {} \n Reactants {}\n Other information: {} \n"\
            .format(self.rxn_id, self.rxn, str(self.reactants), self.others)


class NoKekulizedProducts(ReactionFailed):
    def __init__(self, product_strings: List[str], *args, **kwargs):
        super(NoKekulizedProducts, self).__init__(*args, **kwargs)
        self.products = product_strings

    def __str__(self):
        return super(NoKekulizedProducts, self).__str__()+f"\n Products set : {self.products}"


class ProbablyBadHeteroatomsError(Exception):
    def __init__(self, compound: str, features: np.ndarray, feature_names):
        self.smiles = compound
        self.features = features
        self.feature_names = feature_names

    def __str__(self):
        return super(ProbablyBadHeteroatomsError, self).__str__()+\
               " Failed to compute the descriptors for the compound: {} \nValues: {}\nNames: {}".format(
                   self.smiles,
                   self.features,
                   self.feature_names
               )


def print_verbose(message: str, verbosity: int, verbosity_thr: int, **kwargs):
    if verbosity >= verbosity_thr:
        print(message, **kwargs)


def pd_reset_index(array: pd.DataFrame):
    array.index = pd.RangeIndex(stop=array.shape[0])    # just changes up the index to a new one