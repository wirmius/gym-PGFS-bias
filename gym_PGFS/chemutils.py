import os
import re
import pandas as pd
import numpy as np

from multiprocessing import Pool
from collections import namedtuple
from functools import partial, reduce
from typing import List, Union, Tuple, Callable, Dict

#from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.DataStructs import DiceSimilarity

from rdkit import RDLogger

from rdkit.Chem.rdChemReactions import ChemicalReaction

import pickle

from .config import load_config
from .utils import print_verbose, shelf_directory, ensure_dir_exists, pd_reset_index
from .datasets import load_reactants, load_reactions, extract_templates, \
    get_fingerprint_fn, get_distance_fn, normalize_fingerprints, normalized_fp_fun
from .chemwrapped import ChemMolFromSmilesWrapper, ChemMolToSmilesWrapper, \
    ReactionFromSMARTSAll, ChemMolToSMARTS, evaluate_reactivity




class ChemWorld(object):
    '''
    A class that can provide all the possible reactants and reaction templates.
    '''
    def __init__(self, comp_dir: Union[str, os.PathLike],
                 config: Union[str, os.PathLike, Dict] = None,
                 force_prepro = False):
        super(ChemWorld, self).__init__()

        if isinstance(config, str) or isinstance(config, os.PathLike) or config is None:
            self.config = load_config(config)
        else:
            self.config = config

        self.DATA_DIR = os.path.abspath(comp_dir)
        self.CP_DIR = os.path.join(self.DATA_DIR, self.config['chem_world']['checkpoints']['subdir_prefix'])
        self.verbosity = self.config['preprocessing']['verbosity']

        self.reactants, self.templates, self.rxn = None, None, None
        self.__afps, self.__ofps = None, None

        # initialize variables for normalization of the fingerprints if requested
        if self.config['chem_world']['action_fingerprints']['normalize']:
            self.__afps_mean, self.__afps_stds = None, None
        if self.config['chem_world']['observation_fingerprints']['normalize']:
            self.__ofps_mean, self.__ofps_stds = None, None

        # try loading the latest checkpoint
        if not force_prepro:
            try:
                self.load_checkpoint()
            except AttributeError:
                print_verbose("Could not load a checkpoint, proceed to preprocess", self.verbosity, 1)
                self.preprocess()
        else:
            print_verbose("Forced preprocess selected, last checkpoint will be shelved upon checkpointing.",
                          self.verbosity, 1)
            self.preprocess()

    def deploy(self):
        '''
        Applies optimizations, that would enable faster functioning of the fingerprints, etc.
        Should not do any preprocessing or changes once that is done.
        Returns
        -------
        '''
        # in order to maintain consistency of loc and iloc
        # the templates and reactants can be reindexed safely
        # FIXME: if loc and iloc diverge problems arise with
        #  get template functions, etc.
        pd_reset_index(self.reactants)
        pd_reset_index(self.templates)
        self.__afps = np.vstack(self.reactants[self.action_fp].to_list())
        self.__ofps = np.vstack(self.reactants[self.observation_fp].to_list())

        # initialize the normalization information for the fingerprints if requested
        if self.config['chem_world']['action_fingerprints']['normalize']:
            self.__afps_mean = self.__afps.mean(axis=0)
            self.__afps_stds = self.__afps.std(axis=0)
            self.__afps = normalize_fingerprints(self.__afps,
                                                 self.__afps_mean,
                                                 self.__afps_stds,)
        if self.config['chem_world']['observation_fingerprints']['normalize']:
            self.__ofps_mean = self.__ofps.mean(axis=0)
            self.__ofps_stds = self.__ofps.std(axis=0)
            self.__ofps = normalize_fingerprints(self.__ofps,
                                                 self.__ofps_mean,
                                                 self.__ofps_stds,)


        print_verbose(f"ChemWorld {self} has been deployed.", self.verbosity, 1)
        return

    def action_fps_numpy(self, subset: pd.Series) -> np.ndarray:
        # expects a boolean Series indexed similar to reactant
        # aka output of get_compatible_reactants
        subset = subset.to_numpy()
        return self.__afps[subset, :]

    def observation_fps_numpy(self, subset: pd.Series) -> np.ndarray:
        # expects a boolean Series indexed similar to reactant
        # aka output of get_compatible_reactants
        subset = subset.to_numpy()
        return self.__ofps[subset, :]

    # functions for fingerprint difference computation
    @property
    def action_fp_dist_fn(self):
        return get_distance_fn(self.action_fp_diff)

    @property
    def observation_fp_dist_fn(self):
        return get_distance_fn(self.observation_fp_diff)

    # names of the fingerprint distance functions
    @property
    def action_fp_diff(self):
        return self.config['chem_world']['action_fingerprints']['comparison']

    @property
    def observation_fp_diff(self):
        return self.config['chem_world']['observation_fingerprints']['comparison']

    @property
    def action_fp(self):
        return self.config['chem_world']['action_fingerprints']['type']

    @property
    def observation_fp(self):
        return self.config['chem_world']['observation_fingerprints']['type']

    @property
    def action_fp_fn(self):
        fn, params = get_fingerprint_fn(self.config['chem_world']['action_fingerprints']['type'])
        fn = partial(fn, **params)
        if self.config['chem_world']['action_fingerprints']['normalize']:
            # wrap in a normalization wrapper with the mean and stds supplied
            fn = partial(normalized_fp_fun, fn = fn, means = self.__afps_mean, stds = self.__afps_stds)
        return fn

    @property
    def observation_fp_fn(self):
        fn, params = get_fingerprint_fn(self.config['chem_world']['observation_fingerprints']['type'])
        fn = partial(fn, **params)
        if self.config['chem_world']['observation_fingerprints']['normalize']:
            # wrap in a normalization wrapper with the mean and stds supplied
            fn = partial(normalized_fp_fun, fn = fn, means = self.__ofps_mean, stds = self.__ofps_stds)
        return fn

    def save_checkpoint(self, fmt: str = "pickle"):
        checkpoint_cfg = self.config['chem_world']['checkpoints']

        # ensure that the directory exists, checkpoint the current last save if ti does
        if ensure_dir_exists(self.CP_DIR):
            shelf_directory(self.CP_DIR)

        # save to corresponding files
        if fmt == "pickle":
            with open(os.path.join(
                    self.CP_DIR,
                    checkpoint_cfg['templates']), 'wb') as f:
                pickle.dump(self.templates, f)

            print_verbose(f"Templates dumped to {checkpoint_cfg['templates']}", self.verbosity, 2)
            with open(os.path.join(
                    self.CP_DIR,
                    checkpoint_cfg['reactions']), 'wb') as f:
                pickle.dump(self.rxn, f)
            print_verbose(f"Reactions dumped to {checkpoint_cfg['reactions']}", self.verbosity, 2)
            with open(os.path.join(
                    self.CP_DIR,
                    checkpoint_cfg['reactants']), 'wb') as f:
                pickle.dump(self.reactants, f)
            print_verbose(f"Reactants dumped to {checkpoint_cfg['reactants']}", self.verbosity, 2)
        else:
            raise AttributeError(f"Invalid argument type supplied to the checkpoint function : {fmt}")

        print_verbose(f"Checkpoint complete. ({self.CP_DIR})", self.verbosity, 1)

    def load_checkpoint(self):
        if not os.path.exists(self.CP_DIR):
            raise AttributeError(f"Cant load the checkpoint as the directory does not exists... ({self.CP_DIR})")
        checkpoint_cfg = self.config['chem_world']['checkpoints']

        with open(os.path.join(
                self.CP_DIR,
                checkpoint_cfg['templates']), 'rb') as f:
            self.templates = pickle.load(f)

        print_verbose(f"Templates loaded from {checkpoint_cfg['templates']}", self.verbosity, 2)
        with open(os.path.join(
                self.CP_DIR,
                checkpoint_cfg['reactions']), 'rb') as f:
            self.rxn = pickle.load(f)
        print_verbose(f"Reactions loaded from {checkpoint_cfg['reactions']}", self.verbosity, 2)
        with open(os.path.join(
                self.CP_DIR,
                checkpoint_cfg['reactants']), 'rb') as f:
            self.reactants = pickle.load(f)
        print_verbose(f"Reactants loaded from {checkpoint_cfg['reactants']}", self.verbosity, 2)

        print_verbose("Loaded checkpoint.", self.verbosity, 1)

    def preprocess(self):
        '''
        Loads and preprocesses reactants and reactions according to the configuration.

        Returns
        -------
        '''
        prepro_config = self.config['preprocessing']
        verbosity = prepro_config['verbosity']

        print_verbose("Loading reactions...", verbosity, 1)
        rxn_config = prepro_config['reactions']
        self.rxn = load_reactions(
            rxn_type=rxn_config['source'],
            max_templates=rxn_config['max_templates'],
            max_ants=rxn_config['max_reactants'],
            min_ants=rxn_config['min_reactants'],
            verbosity=verbosity
        )

        print_verbose("Loading templates...", verbosity, 1)
        self.templates = extract_templates(self.rxn)

        print_verbose("Loading reactants...", verbosity, 1)
        rcn_config = prepro_config['reactants']
        self.reactants = load_reactants(
            reactants_type=rcn_config['source'],
            raw_templates_df=self.templates,
            fingerprints=rcn_config['fingerprints'],
            max_reactants=rcn_config['max_reactants'],
            max_heavy_atoms=rcn_config['max_heavy_atoms'],
            verbosity=verbosity,
            n_workers=prepro_config['compute']['n_workers'],
            n_threads=prepro_config['compute']['n_threads_per_worker'],
            max_mem=prepro_config['compute']['max_mem']
        )

        # now apply filter them and ready
        self.restat()
        filter_config = prepro_config['filter']
        if filter_config['perform']:
            self.filter(filter_config)
        else:
            print_verbose(f"Filtering is not perofrmed by default.", verbosity, 1)

        print_verbose("Preprocessing complete.", verbosity, 1)

    def restat(self):
        '''
        Compute or recompute the statistics used for filtering.
        Returns
        -------

        '''
        reactants = self.reactants
        templates = self.templates
        rxn = self.rxn

        # drop the stat columns if any
        if 'n_templates' in reactants:
            del reactants['n_templates']
        if 'n_reactants' in templates:
            del templates['n_reactants']
        if 'n_rcn_low' in rxn:
            del rxn['n_rcn_low']
            del rxn['n_rcn_total']

        # compute how many
        reactivity_cols = filter(lambda a: a.startswith("react_"), reactants.columns)
        stat_rcn = reactants[reactivity_cols].sum(axis=1)  # the number of templates that each reagent fits
        reactants['n_templates'] = stat_rcn

        # compute n_reactants for templates
        # sum up all the reactivity columns
        temp_stats = reactants[[a for a in reactants.columns if a.startswith("react_")]].sum(axis=0)
        temp_stats.name = "n_reactants"
        tstats = pd.merge(temp_stats.to_frame(), temp_stats.index.to_series(name='cname'), right_index=True, left_index=True)

        decode_rxn_id = lambda cname: list(map(int, cname[6:].split('_')))[0]
        decode_rcn_id = lambda cname: list(map(int, cname[6:].split('_')))[1]
        tstats['rxn_id'] = tstats['cname'].apply(decode_rxn_id).astype(int)
        tstats['rcn_id'] = tstats['cname'].apply(decode_rcn_id).astype(int)

        # now perform a join to determine every templates accessible reactants
        templates = pd.merge(templates, tstats[['rxn_id', 'rcn_id', 'n_reactants']],
                             left_on=['rxn_id', 'rcn_id'],
                             right_on=['rxn_id', 'rcn_id'],
                             how='left'
                             )


        # now compute the same for the reactions
        rxn_stats = templates.groupby('rxn_id')['n_reactants'].min()
        rxn_stats_total = templates.groupby('rxn_id')['n_reactants'].sum()

        rxn['n_rcn_low'] = rxn_stats
        rxn['n_rcn_total'] = rxn_stats_total

        self.rxn = rxn
        self.templates = templates
        self.reactants = reactants

        return

    def filter(self,
               f_cfg: Dict = None):
        '''
        This function is for mutual filtering of reactions and reactants. Make sure to run restat() before
        Parameters
        ----------
        f_cfg

        Returns
        -------

        '''
        if not f_cfg:
            f_cfg = self.config['preprocessing']['filter']

        reactants = self.reactants
        templates = self.templates
        rxn = self.rxn
        verbosity = self.verbosity

        min_reactants_per_template = f_cfg['min_reactants_per_template']
        min_template_compatible = f_cfg['min_template_compatible']
        min_reactants_per_reaction = f_cfg['min_reactants_per_reaction']

        # filter down the reactants
        print_verbose(f"n reactants before filtering: {len(reactants)}", verbosity, 1)
        reactants = reactants[reactants['n_templates'] >= min_template_compatible]
        print_verbose(f"n reactants after filtering: {len(reactants)}", verbosity, 1)

        # filter down reactions and templates
        print_verbose(f"n templates before filtering: {len(templates)}", verbosity, 1)
        rxn_drop_index = rxn[rxn['n_rcn_low'] < min_reactants_per_template].index.tolist()
        # now filter out the corresponding templates
        rxn = rxn[rxn['n_rcn_low'] >= min_reactants_per_template]
        print_verbose(f"The following templates are dropped: "+str(templates[templates['rxn_id'].apply(lambda t: t in rxn_drop_index)]),
                      verbosity, 1)
        templates = templates[templates['rxn_id'].apply(lambda t: t not in rxn_drop_index)]
        # done
        print_verbose(f"n templates after filtering: {len(templates)}", verbosity, 1)

        '''# mark templates to be removed
        templates = pd.merge(templates, rxns_to_kill, how='left', left_on='rxn_id', right_index=True)
        templates['tmp_merge_col'].fillna(False)

        # no need to reindex as the features are iloced anyways


        # iterate over the templates, compute how many reactants does each one have in the current dataset
        decode = lambda cname: list(map(int, cname[6:].split('_')))
        lookup = lambda idx: templates.groupby('rxn_id').groups[idx[0]].to_list()
        encode = lambda idx: 'react_{}_{}'.format(templates.loc[idx, 'rxn_id'], templates.loc[idx, 'rcn_id'])



        # now groupby and min to determine the lowest template reactant counts
        rxn_stats = templates.groupby('rxn_id')[''].min()


        # now collect the indices of all templates that do not have enough reactants
        # as well as those that have the other reactant in one of those indices
        keepind = reduce(lambda l, a: l + a,  # if not print(l, a) else l + a,    # concatenate lists of indices
                         temp_stats[temp_stats >= min_reactants_per_template].index
                         .to_series().apply(decode).apply(lookup).to_list(), [])
        # uniquify
        keepind = list(set(keepind))
        # compute indices that we leave intact
        dropind = [c for c in templates.index.to_list() if c not in keepind]
        # also encode those indices to drop columns in reactants
        drop_rcn_cols = list(map(encode, dropind))

        # now apply everything
        templates = templates.loc[keepind].sort_values(['rxn_id', 'rcn_id'])
        templates.index = pd.RangeIndex(stop=templates.shape[0])  # reindex it for proper work with the environment code
        reactants = reactants.drop(drop_rcn_cols,
                                   axis=1)  # now also remove templates that are belonging to the reactions that have deleted templates

        # drop some optional messages'''
        print_verbose("Number of templates passed: {}.\n".format(len(templates.index)), verbosity, 1)

        self.templates = templates
        self.reactants = reactants
        self.rxn = rxn

        return

    def remove_reaction(self, rxn_idx):
        '''
        Remove one reaction by index and all the associated templates.
        Parameters
        ----------
        rxn_idx
            the index of a reaction
        Returns
        -------

        '''
        self.rxn = self.rxn[self.rxn.index != rxn_idx]
        self.templates = self.templates[self.templates['rxn_id'] != rxn_idx]
        return

    def evaluate_reactivity_mask(self, compound, dtype=np.float32) -> np.ndarray:
        ''' What this function does is to evaluate reactivity in various roles in various reactions.
        Returned is a bit vector denoting available reactions.
        Fits both strings and mols.
        '''
        # TODO: filter for potential unintended intramolecular reactions by checking whether the compound satisfies
        #   requirements for all of the components of some reactions
        if isinstance(compound, str):
            target_mol = ChemMolFromSmilesWrapper(compound)
        else:
            target_mol = compound
        return self.templates["mol"].apply(
            lambda q: True if len(target_mol.GetSubstructMatches(q, True)) != 0 else False
        ).to_numpy(dtype=dtype)

    def get_reaction_by_index(self, rxn_idx: int) -> ChemicalReaction:
        # be careful and only ever provide a copy of the chemical reaction
        return ChemicalReaction(self.rxn.loc[rxn_idx]['robject'])

    def get_reaction_index_from_template_index(self, tmp_idx: int) -> int:
        return self.templates.loc[tmp_idx]['rxn_id']

    def get_templates_for_reaction_index(self, rxn_idx) -> List:
        return self.templates.groupby('rxn_id').groups[rxn_idx].to_list()

    def get_template_order_from_template_index(self, tmp_idx):
        return self.templates.loc[tmp_idx]['rcn_id']

    def get_compatible_reactant_subset(self, tmp_idx: int) -> pd.Series:
        # provides the reactants compatible with the given reactant
        return self.reactants[
            'react_{}_{}'.format(self.templates.loc[tmp_idx]['rxn_id'],
                                 self.templates.loc[tmp_idx]['rcn_id'])]

    def get_molecular_fingerprint(self, smi: str, role: str) -> np.ndarray:
        return self.fp_funs[role](smi)

    @property
    def action_fp_len(self):
        return self.reactants[self.action_fp].iloc[0].shape[0]

    @property
    def observation_fp_len(self):
        return self.reactants[self.observation_fp].iloc[0].shape[0]

    @property
    def template_count(self):
        return len(self.templates)

    def get_reactant_smiles(self, idx):
        return self.reactants['smiles'][idx]

    def get_reaction_str_repr(self, rxn_idx):
        if "name" in self.rxn:
            return self.rxn["name"][rxn_idx]
        else:
            return rxn_idx




if __name__ == "__main__":
    print("chemutils.py shouldnt be run directly.")
    exit(-1)
else:
    pass
    # log to file instead of console
    RDLogger.DisableLog("rdApp.info")
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    # RDLogger.AttachFileToLog("rdApp.info", os.path.join(LOG_DIR, "chemutils_i.log"))
    # RDLogger.AttachFileToLog("rdApp.warning", os.path.join(LOG_DIR, "chemutils_w.log"))   <<-- doesnt work :\
