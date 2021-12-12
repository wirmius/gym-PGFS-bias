import sys

import numpy as np
import pandas as pd

from gym_PGFS.external.pollo1060.pipeliner_light.pipelines import ClassicPipe
from gym_PGFS.external.pollo1060.pipeliner_light.smol import SMol
from .chemwrapped import Mol, ChemMolToSmilesWrapper

import os
from typing import Union, Callable
from functools import partial

from .datasets import get_fingerprint_fn, get_distance_fn
from .utils import ProbablyBadHeteroatomsError

import abc
from typing import Tuple, Dict

from .constants import PGFS_MODULE_ROOT


# def clamp_and_scale_score(m, score: Callable, low, high, invert=False) -> float:
#     # apply the function
#     f = score(m)
#
#     # clamp the result and scale
#     f = (max(low, min(high, f)) - low)/(high-low)
#
#     return f - 1


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


class ScoringFunctionPGFS(abc.ABC):

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


class PipelinerScorepCHEMBL(ScoringFunctionPGFS):

    def __init__(self, **kwargs):
        # the name indicates the name of one of the three proteins used,
        # possible values: hiv_ccr5, hiv_int, hiv_rt
        self.name = kwargs['name']

        # the transform that we apply
        # possible values: none, norm, clamp, norm_clamp
        self.transform = kwargs['transform']

        # load the models
        self.pipeline = ClassicPipe.load(os.path.join(PGFS_MODULE_ROOT, f'external/pollo1060/Models/{self.name}'))
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

#
# ccr5_pipe = ClassicPipe.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                           'external/pollo1060/Models/hiv_ccr5'))
# int_pipe = ClassicPipe.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                          'external/pollo1060/Models/hiv_int'))
# rt_pipe = ClassicPipe.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external/pollo1060/Models/hiv_rt'))
#
#
# def ccr5_score_pollo_1060(m: Union[Mol, str]) -> float:
#     '''
#
#     Parameters
#     ----------
#     m
#         A mol or an str
#     Returns
#     -------
#
#     '''
#     sm = SMol(m)
#     sm.featurize(ccr5_pipe.features)
#     if np.any(np.isinf(sm.features_values)):
#         # in an attempt to catch the elusive bug with two inf values at pos 6 and 8
#         raise ProbablyBadHeteroatomsError(sm.smiles,
#                                           sm.features_values,
#                                           sm.features_names)
#         #print(sm.features_values)
#         #print(sm.features_names)
#         #print(m)
#         #print(ChemMolToSmilesWrapper(sm.rmol))
#     return ccr5_pipe.predict_vector(sm.features_values)
#
#
# def hiv_int_score_pollo_1060(m: Union[Mol, str]) -> float:
#     '''
#
#     Parameters
#     ----------
#     m
#         A mol or an str
#     Returns
#     -------
#
#     '''
#     sm = SMol(m)
#     sm.featurize(int_pipe.features)
#     if np.any(np.isinf(sm.features_values)):
#         # in an attempt to catch the elusive bug with two inf values at pos 6 and 8
#         raise ProbablyBadHeteroatomsError(sm.smiles,
#                                           sm.features_values,
#                                           sm.features_names)
#     return int_pipe.predict_vector(sm.features_values)
#
#
# def hiv_rt_score_pollo_1060(m: Union[Mol, str]) -> float:
#     '''
#
#     Parameters
#     ----------
#     m
#         A mol or an str
#     Returns
#     -------
#
#     '''
#     sm = SMol(m)
#     sm.featurize(rt_pipe.features)
#     if np.any(np.isinf(sm.features_values)):
#         # in an attempt to catch the elusive bug with two inf values at pos 6 and 8
#         raise ProbablyBadHeteroatomsError(sm.smiles,
#                                           sm.features_values,
#                                           sm.features_names)
#     return rt_pipe.predict_vector(sm.features_values)
#
#
# # initialize some globals for the toy score function
# fp_fun, fp_params = get_fingerprint_fn('ECFP_4_1024')
# fp_distance = get_distance_fn('dice')
#
# #tetra_smiles = 'CC1(C(CC(=O)C3=C(C4(C(CC31)C(C(=C(C4=O)C(=O))O))O)O)O)O'
# tetra_smiles = 'C1(=C2C(CCC1)C(C(CC2=O)O)(C)O)O'
# tetra_fp = fp_fun(tetra_smiles, **fp_params)
#
# def toy_set_score(m: Union[Mol, str]) -> float:
#     # in the toy task, the purpose is to generate a molecule that is as similar as possible to the
#     # tetracycline-derived structure, which should be achievable given the materials in the toy set
#     fp = fp_fun(m, **fp_params)
#     d = fp_distance(tetra_fp, fp)
#     return d
#
#
# # function, low, high
# scoring_function_registry = {
#     'ccr5_pCHEMBL': [ccr5_score_pollo_1060, {'low': 5, 'high': 10, 'mean': 6, 'std': 1}],
#     'int_pCHEMBL': [hiv_int_score_pollo_1060, {'low': 5, 'high': 10, 'mean': 6, 'std': 1}], # not accurate, needs elaboration
#     'rt_pCHEMBL': [hiv_rt_score_pollo_1060, {'low': 4, 'high': 8, 'mean': 5.93, 'std': 0.43}],
#     'toy_score': [toy_set_score, {'low': 5, 'high': 10, 'mean': 6, 'std': 1}]
# }
#

def get_scoring_function(type: str, **params) -> ScoringFunctionPGFS:
   if type == "pollo1060":
       return PipelinerScorepCHEMBL(**params)
   elif type == "guacamol":
       pass
   elif type == "guacamol_mgenfail":
       pass
   else:
       raise AttributeError(f"Scoring function type {type} doesnt exist.")
