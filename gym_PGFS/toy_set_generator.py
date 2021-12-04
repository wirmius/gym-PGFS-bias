import numpy as np
import pandas as pd
from numpy.random import RandomState
from tqdm import tqdm
from rdkit import RDLogger


from .constants import DOWNLOAD_ROOT
from .chemwrapped import ChemMolFromSmilesWrapper, ChemMolToSmilesWrapper

possible_f_groups = ['O', '(O)','(=O)', '=O', '(=O)O', 'Cl', '=', '=', '=', '=', '=']


def insert_string(basestr: str, insert: str, pos: int) -> str:
    return basestr[:pos] + insert + basestr[pos:]


def generate_fake_mol(rs: RandomState) -> str:
    n_carbons = rs.randint(1, 8)
    n_f_groups = rs.randint(0, min(n_carbons, 4))
    n_branches = rs.randint(0, max(n_carbons-4, 1))# + n_f_groups

    # start with making a backbone
    basestr = ''.join(['C']*n_carbons)

    # introduce the branching
    idx_list = sorted(rs.randint(1, len(basestr), size=n_branches*2).tolist())
    idx_relist = [idx + i for i, idx in enumerate(idx_list)]    # the future index of the parenthesis
    for i, idx in enumerate(idx_relist):
        if i%2 == 0:
            basestr = insert_string(basestr, '(', idx)
        else:
            basestr = insert_string(basestr, ')', idx)


    # introduce the functional groups
    groups = rs.choice(possible_f_groups, n_f_groups, replace=True).tolist()
    locs = rs.randint(1, len(basestr), size=n_f_groups).tolist()
    for g, l in zip(groups, locs):
        if l == -1:
            return None
        basestr = insert_string(basestr, g, l)

    # verify the molecule
    mol = ChemMolFromSmilesWrapper(basestr)
    if mol:
        return ChemMolToSmilesWrapper(mol)
    else:
        return None


def generate_fake_set(n_compounds = 1000) -> pd.Series:

    RDLogger.DisableLog("rdApp.info")
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    randomstate = RandomState(seed=228)
    retset = set({})
    with tqdm(total=n_compounds) as pbar:
        while len(retset) < n_compounds:
            smiles = None
            while not smiles:
                smiles = generate_fake_mol(randomstate)
            retset.add(smiles)
            pbar.n = len(retset)
            pbar.refresh()
    pbar.close()
    return pd.Series(list(retset), name='smiles')


def deploy_fake_set(n_compounds):
    generate_fake_set(n_compounds=n_compounds).to_csv(DOWNLOAD_ROOT+'/toy_reactants.csv')