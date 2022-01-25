import os
import pandas as pd
import numpy as np
import re
from typing import Union, List, Tuple, Callable, Dict

from .constants import DOWNLOAD_ROOT
from .utils import print_verbose, pd_reset_index

from dask import dataframe as dd
from dask.distributed import Client, progress

from functools import partial


from .chemwrapped import ChemMolFromSmilesWrapper, ChemMolToSmilesWrapper, \
    ReactionFromSMARTSAll, ChemMolToSMARTS, evaluate_reactivity, \
    ECFP6_bitvector_numpy, CosDistance, DiceDistance, MolDSetDescriptorsNumpy, \
    ChemGuacamolFilterExtended


def load_toy_reactants_raw() -> pd.DataFrame:
    fname = os.path.join(DOWNLOAD_ROOT, "toy_reactants.csv")
    return pd.read_csv(fname, dtype=str)


def load_USPTO_reactant_set_raw() -> pd.DataFrame:
    fname = os.path.join(DOWNLOAD_ROOT, "uspto_reactants_sorted.txt")
    return pd.read_csv(fname, header=None, names=['smiles'], dtype=str)


def load_Enamines_reactant_set_raw() -> pd.DataFrame:
    fname = os.path.join(DOWNLOAD_ROOT, "enamine_building_blocks.csv")
    return pd.read_csv(fname, header=1, names=['SMILES'], dtype=str).rename(columns={'SMILES': 'smiles'})


def load_toy_reaction_set_raw() -> pd.DataFrame:
    fname = os.path.join(DOWNLOAD_ROOT, "DINGOS_toy_set.csv")
    retval = pd.read_csv(fname, sep=',', header=0, names=['name', 'smarts', 'source'], dtype=str)
    # as all DINGOS reaction templates are intended to be bimolecular
    retval['intra_only'] = False
    # I assume this field means that this reaction is valid only for dimerisation, shouldnt be the case for the DINGOS
    retval['dimer_only'] = False
    # leave count empty
    return retval


def load_USPTO_reaction_set_raw() -> pd.DataFrame:
    # is contained ina  json
    fname = os.path.join(DOWNLOAD_ROOT, "uspto_sm_retro.templates.uspto_sm_.json")
    retval = pd.read_json(fname)
    retval = retval[['reaction_smarts', 'intra_only', 'dimer_only', 'count']]
    retval.rename(columns={'reaction_smarts': 'smarts'}, inplace=True)
    return retval


def load_DINGOS_reaction_set_raw() -> pd.DataFrame:
    fname = os.path.join(DOWNLOAD_ROOT, "rxn_set_DINGOS.txt")
    retval = pd.read_csv(fname, sep='|', header=0, names=['name', 'smarts', 'source'], dtype=str)
    # as all DINGOS reaction templates are intended to be bimolecular
    retval['intra_only'] = False
    # I assume this field means that this reaction is valid only for dimerisation, shouldnt be the case for the DINGOS
    retval['dimer_only'] = False
    # leave count empty
    return retval


reactants_loaders = {
    "USPTO": load_USPTO_reactant_set_raw,
    "Enamine": load_Enamines_reactant_set_raw,
    "toy": load_toy_reactants_raw,
}

reactions_loaders = {
    "USPTO": (load_USPTO_reaction_set_raw, True),
    "DINGOS": (load_DINGOS_reaction_set_raw, False),
    "toy": (load_toy_reaction_set_raw, False),
}

# possible representations of molecules
mol_represenations = [
    "mol",
    "str"
]

# TODO: fix the list of fingerprints, currently all the ECFPs are running at twice the diameter
fingerprint_functions_list = {
    "ECFP_4_1024": (ECFP6_bitvector_numpy, {"radius": 4, "size": 1024}),
    "ECFP_2_1024": (ECFP6_bitvector_numpy, {"radius": 2, "size": 1024}),
    "ECFP_2_512": (ECFP6_bitvector_numpy, {"radius": 2, "size": 512}),
    "ECFP_2_256": (ECFP6_bitvector_numpy, {"radius": 2, "size": 256}),
    "ECFP_2_128": (ECFP6_bitvector_numpy, {"radius": 2, "size": 128}),
    "MolD": (MolDSetDescriptorsNumpy, {}),
    "MACCS_rdkit": (None, {})   # TODO: implement MACCS (although not a priority, they dont work that well in either role)
}


distance_functions = {
    "dice": DiceDistance,
    "cos": CosDistance
}


def normalize_fingerprints(fp: np.ndarray,
                           means: np.ndarray,
                           stds: np.ndarray):
    if fp.ndim == 1:
        return (fp-means)/stds
    elif fp.ndim == 2:
        return (fp-means)/stds
    else:
        raise ValueError("What exactly are you trying to normalize?")


def normalized_fp_fun(mol, fn: Callable, means: np.ndarray, stds: np.ndarray):
    return normalize_fingerprints(fn(mol), means, stds)


def get_distance_fn(type: str) -> Callable:
    return distance_functions[type]


def get_fingerprint_fn(fp_type: str) -> Tuple[Callable, Dict]:
    fn, params = fingerprint_functions_list[fp_type]
    return fn, params


def load_reactions(rxn_type: str,
                   max_templates=100,
                   max_ants=2,
                   min_ants=1,
                   min_count=0,
                   descriptors_to_compute=[],
                   allow_intra_only=False,
                   allow_dimer_only=False,
                   verbosity: int = 1,
                   ) -> pd.DataFrame:
    '''
    Load the manually created templates from DINGOS paper [Button et al, Automated de novo molecular design by hybrid
    machine intelligence and rule driven chemical synthesis].
    --> I think are better suited for the task than the massive amount of autoextracted and arbitrary quality rdchiral
    templates. Also only 64 most common reactions are in there.
    --> roll with top 100 uspto reactions for now
    --> also not the worst idea to store it in a dataframe
    '''
    print_verbose(f"Loading reactions({rxn_type}) from the source...", verbosity, 1)
    # load raw reaction smarts
    lfunc, fix_retro = reactions_loaders[rxn_type]
    rxns = lfunc()

    # drop the intra only and dimer only if needed
    if not allow_intra_only:
        print_verbose("Removing reactions that are intramolecular only.", verbosity, 2)
        rxns = rxns[~rxns['intra_only']]
    if not allow_dimer_only:
        print_verbose("Removing reactions that are dimerization only.", verbosity, 2)
        rxns = rxns[~rxns['dimer_only']]

    # fix retro if needed
    if fix_retro:
        print_verbose("Retrochemical reaction, fixing...", verbosity, 2)
        rxns['smarts'] = rxns['smarts'].apply(lambda rx: '>>'.join(rx.split('>>')[::-1]))

    # sort by 'count' if 'count' is present, otherwise skip
    if 'count' in rxns.columns:
        # since we are here, we can filter reactions by count
        rxns = rxns[rxns['count'] >= min_count]
        rxns.sort_values('count', ascending=False, inplace=True)

    # interpret the reactions with rdkit
    print_verbose("Getting reaction objects...", verbosity, 2)
    rxns['robject'] = rxns['smarts'].apply(lambda rxn_smarts: ReactionFromSMARTSAll(rxn_smarts))
    rxns.dropna(axis=0, inplace=True)  # drop rows with missing values

    # compute number of reactants
    rxns['n_reactants'] = rxns['robject'].apply(lambda rxn: rxn.GetNumReactantTemplates())

    # leave only the reactions that satisfy the criteria
    print_verbose(f"Prefiltering reaction by the reactant number (max: {max_ants}, min: {min_ants})...", verbosity, 2)
    rxns = rxns[rxns['n_reactants'] <= max_ants]
    rxns = rxns[rxns['n_reactants'] >= min_ants]

    # crop off the top n reaction templates
    rxns['template_cumsum'] = rxns['n_reactants'].cumsum()
    rxns = rxns[rxns['template_cumsum'] < max_templates]
    print_verbose(f"Final number of reactants: {len(rxns)}; templates: {rxns['template_cumsum'].iloc[-1]}...", verbosity, 2)

    # TODO: ??
    #  optionally compute the reaction fingerprints

    # reset the index
    pd_reset_index(rxns)

    # done
    return rxns


def extract_templates(rxns_df: pd.DataFrame) -> pd.DataFrame:
    templates_df = pd.DataFrame(columns=['rxn_id', 'rcn_id', 'smarts', 'mol'])
    for i, row in rxns_df.iterrows():
        reactants = row['robject'].GetReactants()
        for t_idx, template in enumerate(reactants):
            templates_df = templates_df.append({'rxn_id': i, 'rcn_id': t_idx, 'smarts': ChemMolToSMARTS(template),
                                                'mol': template}, ignore_index=True)
    # in order to make sure there are no inconsistencies here
    pd_reset_index(templates_df)
    # done
    return templates_df


#### TODO: drop the infs and nans from the descriptors (just in case)
def load_reactants(reactants_type: str,
                   raw_templates_df: pd.DataFrame,
                   fingerprints: List = [],
                   max_reactants: int = -1,
                   max_heavy_atoms: int = -1,
                   verbosity: int = 1,
                   batch_size: int = 5000,
                   n_workers: int = 8,      # useful since the rdkit functions most likely employ GIL lock
                   n_threads: int = 1,
                   max_mem: int = 24,
                   ) -> pd.DataFrame:
    '''
    immediately catalogues the reactants according to their reactivity according to the templates already loaded.
    this can allow a faster lookup of reactants

    so the reactants set should be a pandas df with
    'smiles':str,
    'mol':rdkit.Mol,
    'valid':bool (whether the molecule
    passed all of the filters/checks from rdkit during smiles conversion,
    'react_{}_{}':bool * all reactant in all reactions, which stands for whether they match the pattern for the reactant
    #2 in reaction #1.

    this way we can also then filter out all the reactants that match no pattern and run some stats on the reactants and
    templates

    '''
    assert raw_templates_df is not None

    # load the raw reactants (should have only one column - smiles)
    print_verbose(f"Loading reactant ({reactants_type})...", verbosity, 1)
    reactants = reactants_loaders[reactants_type]()

    print_verbose(f"Raw reactants loaded, n = {len(reactants)}", verbosity, 2)

    print_verbose(f"Filtering reactants using the guacamol routines and selecting only the unique reactants...", verbosity, 2)

    reactants['smiles'] = reactants['smiles'].apply(ChemGuacamolFilterExtended)
    reactants = reactants.drop_duplicates(subset='smiles').dropna(subset=['smiles'])
    # print(reactants['smiles'].apply(lambda a: type(a)).drop_duplicates())
    # print(reactants.smiles[96])

    # first trim down based on the number of heavy atoms
    if max_heavy_atoms > 0:
        reactants['n_heavy_atoms'] = reactants['smiles'].apply(
            lambda smi: len(re.findall('[A-Z]|[c,n,o]', smi))
        )
        reactants = reactants[reactants['n_heavy_atoms']< max_heavy_atoms]
        print_verbose(f"Discarded reactants over the size of {max_heavy_atoms} atoms, reactants left {len(reactants)}", verbosity, 2)

    # if there are too many reactants, cut them down
    if max_reactants > 0:
        reactants = reactants.iloc[0:max_reactants]

    # prepare the dataframe for processing
    n_batches = len(reactants.index) // batch_size + 1
    reactants = dd.from_pandas(reactants, chunksize=batch_size)
    print_verbose(f"Raw reactants loaded, total: {len(reactants.index)}\n", verbosity, 2)

    # now go lazily to describe the following computation to dask
    # get Mol object
    mols = reactants['smiles'].apply(ChemMolFromSmilesWrapper, meta=('mol', object)).dropna() # immediately discard all the null molecules

    # use them and the templates frame to compute the reactivities
    reactivities = [
        mols.apply(evaluate_reactivity, template_mol=row['mol'], meta=(
            'react_{}_{}'.format(row['rxn_id'], row['rcn_id']), bool)
                   )
        for index, row in raw_templates_df.iterrows()]

    # use mols again to compute the fingerprints
    fp_cols = []
    for fp in fingerprints:
        func, kwargs = get_fingerprint_fn(fp)
        fp_cols.append(
            mols.apply(func, meta=(fp, object), **kwargs)
        )

    resmiles = mols.apply(ChemMolToSmilesWrapper, meta=('smiles', str)).to_frame()
    # axis=1 not needed as we deal with series in all cases

    resmiles = dd.merge(resmiles,
                        mols.apply(lambda mol: mol.GetNumAtoms(), meta=('n_heavy', int)),
                        right_index=True,
                        left_index=True,
                        how='left')

    # now join the reactivities
    for react in reactivities:
        resmiles = dd.merge(resmiles, react, left_index=True, right_index=True)

    # add the fingerprints as well
    for fp in fp_cols:
        resmiles = dd.merge(resmiles, fp, left_index=True, right_index=True)

    # now do the computation
    with Client(memory_limit=str(max_mem//n_workers)+"GB", n_workers=n_workers, threads_per_worker=n_threads, processes=False) as client:
        print_verbose(f"Computing using {client}...", verbosity, 1)
        ret_futures = client.persist(resmiles)
        progress(ret_futures)
        ret_df = client.gather(ret_futures).compute()

    # reset the index
    pd_reset_index(ret_df)

    # done
    return ret_df
