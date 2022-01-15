import sys

import numpy as np
import pandas as pd

from gym_PGFS.external.pollo1060.pipeliner_light.pipelines import ClassicPipe
from gym_PGFS.external.pollo1060.pipeliner_light.smol import SMol
from gym_PGFS.chemwrapped import Mol, ECFP6_bitvector_numpy, ChemMolToSmilesWrapper

import os
from typing import Union, Callable
from functools import partial
import pickle

from gym_PGFS.datasets import get_fingerprint_fn, get_distance_fn
from gym_PGFS.utils import ProbablyBadHeteroatomsError

import abc
from enum import Enum
from typing import Tuple, Dict

from gym_PGFS.constants import PGFS_MODULE_ROOT


def normalize_score(score, mean, std) -> float:
    # normalize the score
    f = (score - mean)/std
    return f


def clamp_score(score, low, high) -> float:
    f = np.clip(score, low, high)
    return f


def normalize_and_clamp_score(score, mean, std, low, high) -> float:
    f = clamp_score(normalize_score(score, mean, std), low, high)
    return f


class ScorerPGFS(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_id(self) -> str:
        '''

        Returns
        -------
            detailed name of the scoring function
        '''
        pass

    @abc.abstractmethod
    def score(self, mol: Union[Mol, str]) -> float:
        '''
        Returns the fully processed score, suitable to train an env on.

        Parameters
        ----------
        mol
            molecule to score
        Returns
        -------
            score to be returned by the environment, ultimately
        '''
        pass

    @abc.abstractmethod
    def present_score(self, score: float) -> float:
        '''

        Parameters
        ----------
        score

        Returns
        -------

        '''
        pass


class PipelinerScorer(ScorerPGFS):

    def __init__(self, **kwargs):
        # the name indicates the name of one of the three proteins used,
        # possible values: hiv_ccr5, hiv_int, hiv_rt
        self.name = kwargs['name']

        # the transform that we apply
        # possible values: none, norm, clamp, norm_clamp
        self.transform = kwargs['transform']

        # load the models
        self.pipeline = ClassicPipe.load(os.path.join(PGFS_MODULE_ROOT, f'external/pollo1060/Models/{self.name}'))
        # TODO: turn transforms into a list
        if self.transform == "none":
            self._transform = lambda a: a
        elif self.transform == "norm":
            self._load_means_stds()
            self._transform = partial(normalize_score,
                                      mean=self.mean, std=self.std)
        elif self.transform == "clamp":
            self._load_low_high(kwargs)
            self._transform = partial(clamp_score,
                                      low=self.low, high=self.high)
        elif self.transform == "norm_clamp":
            self._load_means_stds()
            self._load_low_high(kwargs)
            self._transform = partial(normalize_and_clamp_score,
                                      mean=self.mean, std=self.std,
                                      low=self.low, high=self.high)
        else:
            raise AttributeError(f"invalid configuration value for 'transform' {self.transform}")

    def _load_low_high(self, kwargs: Dict):
        self.low = kwargs['low'] if 'low' in kwargs else sys.float_info.min
        self.high = kwargs['high'] if 'high' in kwargs else sys.float_info.max

    def _load_means_stds(self):
        # load the data for the specific scoring function and evaluate
        source_data = pd.read_csv(os.path.join(PGFS_MODULE_ROOT, f'external/pollo1060/Data/{self.name}.csv'))
        scores = source_data['pChEMBL Value'].to_numpy()
        # get the statistics
        self.mean = scores.mean()
        self.std = scores.std()

    def get_id(self) -> str:
        return "pollo_1060_" + self.name

    def score(self, mol: Union[Mol, str]) -> float:
        sm = SMol(mol)
        sm.featurize(self.pipeline.features)
        if np.any(np.isinf(sm.features_values)):
            # in an attempt to catch the elusive bug with two inf values at pos 6 and 8
            raise ProbablyBadHeteroatomsError(sm.smiles,
                                              sm.features_values,
                                              sm.features_names)
        return self._transform(self.pipeline.predict_vector(sm.features_values))

    def present_score(self, score: float):
        return score * self.std + self.mean if self.transform.startswith("norm") else score


class ScorerModeSelector(Enum):
    OS_MODE = 0
    MCS_MODE = 1
    DCS_MODE = 2


class MGenFailScorer(ScorerPGFS):
    def __init__(self, **kwargs):
        '''
        fp_fun: str,
        prefix='./data/mgenfail_assays',
        dataset = 'CHEMBL1909203',
        transforms: list = [],
        Parameters
        ----------
        fp_fun
        prefix
        dataset
        transforms
            can contain 'center', 'proba'
            'proba' is actually there by default
        kwargs
        '''
        dataset = kwargs['name']
        fp_fun = kwargs['fingerprints_used']
        prefix = kwargs['mgenfail_data_prefix']
        transforms = kwargs['transforms']


        os_model = prefix+'/'+dataset+'/OS_MODEL/model.pkl'
        mcs_model = prefix+'/'+dataset+'/MCS_MODEL/model.pkl'
        dcs_model = prefix+'/'+dataset+'/DCS_MODEL/model.pkl'
        # load all three models
        with open(os_model, 'rb') as f:
            self.os_model = pickle.load(f)
        # load all three models
        with open(dcs_model, 'rb') as f:
            self.dcs_model = pickle.load(f)
        # load all three models
        with open(mcs_model, 'rb') as f:
            self.mcs_model = pickle.load(f)

        # set other parameters
        self.dataset_name = dataset

        # fp fun is expected to take either smiles or Mol
        fn, params = get_fingerprint_fn(fp_fun)
        self.fp_fun = partial(fn, **params)

        # set the mode by default to optimisation score
        self.mode = ScorerModeSelector.OS_MODE

        # initialize transforms
        if 'center' in transforms:
            self._transform = lambda a: a - 0.5
        else:
            self._transform = lambda a: a

    def get_id(self) -> str:
        return self.dataset_name

    def get_mode(self) -> ScorerModeSelector:
        return self.mode

    def set_mode(self, mode: ScorerModeSelector):
        self.mode = mode

    def score(self, mol: Union[Mol, str]) -> float:
        mol_desc = self.fp_fun(mol)
        if not isinstance(mol_desc, np.ndarray):
            raise ValueError(f'Descriptor returned to scoring function is not an ndarray, is of type {type(mol_desc)}')
        elif len(mol_desc.shape) == 1:
            # reshape to let it be used by the random forest if just one sample
            mol_desc = mol_desc[np.newaxis, :]
        # use to predict probabilities to obtain a smoother objective
        if self.mode == ScorerModeSelector.OS_MODE:
            score = self.os_model.predict_proba(mol_desc)[0, 1]
        elif self.mode == ScorerModeSelector.MCS_MODE:
            score = self.mcs_model.predict_proba(mol_desc)[0, 1]
        elif self.mode == ScorerModeSelector.DCS_MODE:
            score = self.dcs_model.predict_proba(mol_desc)[0, 1]
        return self._transform(score)

    def present_score(self, score: float) -> float:
        return score


def get_scoring_function(type: str, **params) -> ScorerPGFS:
   if type == "pollo1060":
       return PipelinerScorer(**params)
   elif type == "guacamol":
       raise NotImplemented
   elif type == "guacamol_mgenfail":
       return MGenFailScorer(**params)
   else:
       raise AttributeError(f"Scoring function type {type} doesnt exist.")
