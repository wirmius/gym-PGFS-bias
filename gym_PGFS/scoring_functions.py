import numpy as np

from gym_PGFS.external.pollo1060.pipeliner_light.pipelines import ClassicPipe
from gym_PGFS.external.pollo1060.pipeliner_light.smol import SMol
from .chemwrapped import Mol, ChemMolToSmilesWrapper

import os
from typing import Union, Callable
from functools import partial

from .datasets import get_fingerprint_fn, get_distance_fn
from .utils import ProbablyBadHeteroatomsError

ccr5_pipe = ClassicPipe.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'external/pollo1060/Models/hiv_ccr5'))
int_pipe = ClassicPipe.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'external/pollo1060/Models/hiv_int'))
rt_pipe = ClassicPipe.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external/pollo1060/Models/hiv_rt'))


def clamp_and_scale_score(m, score: Callable, low, high, invert=False) -> float:
    # apply the function
    f = score(m)

    # clamp the result and scale
    f = (max(low, min(high, f)) - low)/(high-low)

    return f - 1


def normalize_score(m, score: Callable, mean, std, invert=False) -> float:
    # apply the function
    f = score(m)

    # normalize the score
    f = (f - mean)/std

    return f


def ccr5_score_pollo_1060(m: Union[Mol, str]) -> float:
    '''

    Parameters
    ----------
    m
        A mol or an str
    Returns
    -------

    '''
    sm = SMol(m)
    sm.featurize(ccr5_pipe.features)
    if np.any(np.isinf(sm.features_values)):
        # in an attempt to catch the elusive bug with two inf values at pos 6 and 8
        raise ProbablyBadHeteroatomsError(sm.smiles,
                                          sm.features_values,
                                          sm.features_names)
        #print(sm.features_values)
        #print(sm.features_names)
        #print(m)
        #print(ChemMolToSmilesWrapper(sm.rmol))
    return ccr5_pipe.predict_vector(sm.features_values)


def hiv_int_score_pollo_1060(m: Union[Mol, str]) -> float:
    '''

    Parameters
    ----------
    m
        A mol or an str
    Returns
    -------

    '''
    sm = SMol(m)
    sm.featurize(int_pipe.features)
    if np.any(np.isinf(sm.features_values)):
        # in an attempt to catch the elusive bug with two inf values at pos 6 and 8
        raise ProbablyBadHeteroatomsError(sm.smiles,
                                          sm.features_values,
                                          sm.features_names)
    return int_pipe.predict_vector(sm.features_values)


def hiv_rt_score_pollo_1060(m: Union[Mol, str]) -> float:
    '''

    Parameters
    ----------
    m
        A mol or an str
    Returns
    -------

    '''
    sm = SMol(m)
    sm.featurize(rt_pipe.features)
    if np.any(np.isinf(sm.features_values)):
        # in an attempt to catch the elusive bug with two inf values at pos 6 and 8
        raise ProbablyBadHeteroatomsError(sm.smiles,
                                          sm.features_values,
                                          sm.features_names)
    return rt_pipe.predict_vector(sm.features_values)


# initialize some globals for the toy score function
fp_fun, fp_params = get_fingerprint_fn('ECFP_4_1024')
fp_distance = get_distance_fn('dice')

#tetra_smiles = 'CC1(C(CC(=O)C3=C(C4(C(CC31)C(C(=C(C4=O)C(=O))O))O)O)O)O'
tetra_smiles = 'C1(=C2C(CCC1)C(C(CC2=O)O)(C)O)O'
tetra_fp = fp_fun(tetra_smiles, **fp_params)

def toy_set_score(m: Union[Mol, str]) -> float:
    # in the toy task, the purpose is to generate a molecule that is as similar as possible to the
    # tetracycline-derived structure, which should be achievable given the materials in the toy set
    fp = fp_fun(m, **fp_params)
    d = fp_distance(tetra_fp, fp)
    return d


# function, low, high
scoring_function_registry = {
    'ccr5_pCHEMBL': [ccr5_score_pollo_1060, {'low': 5, 'high': 10, 'mean': 6, 'std': 1}],
    'int_pCHEMBL': [hiv_int_score_pollo_1060, {'low': 5, 'high': 10, 'mean': 6, 'std': 1}], # not accurate, needs elaboration
    'rt_pCHEMBL': [hiv_rt_score_pollo_1060, {'low': 4, 'high': 8, 'mean': 5.93, 'std': 0.43}],
    'toy_score': [toy_set_score, {'low': 5, 'high': 10, 'mean': 6, 'std': 1}]
}


def get_scoring_function(id: str, scoring_transform = 'none') -> Callable:
    fn, props = scoring_function_registry[id]
    low, high, mean, std = props['low'], props['high'], props['mean'], props['std']
    if scoring_transform=='scale':
        return partial(clamp_and_scale_score, score=fn, low=low, high=high)
    elif scoring_transform=='norm':
        return partial(normalize_score, score=fn, mean=mean, std=std)
    else:
        return fn
