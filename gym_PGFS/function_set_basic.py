import os

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


# import the clean-up objects used fo molecules in the score functions
from .external.pollo1060.pipeliner_light.smol import SMol

from .chemutils import ChemWorld
from .chemwrapped import CheckMol, ChemMolFromSmilesWrapper, ChemMolToSmilesWrapper, SmilesToSVG
from .utils import ReactionFailed, NoKekulizedProducts, ensure_dir_exists
from .external.svg_stack import svg_stack as ss

from sklearn.neighbors import NearestNeighbors

from typing import List, Tuple, Dict

from gym.spaces import Tuple as SpaceDict, Discrete, Box
from gym.error import InvalidAction, ResetNeeded

import numpy as np
from numpy.random import RandomState

from .forward_model import Reaction_Model

# import stuff necessary for rendering
from .constants import DEFAULT_MOL_WIDTH, DEFAULT_MOL_HEIGHT, DEFAULT_RXN_TEXT_HEIGHT, DEFAULT_SCORE_TEXT_HEIGHT, \
    DEFAULT_ARROW_HEIGHT, \
    additive_template, additive_template_R2, \
    swag_arrow, swag_score, swag_rxn_text

class Default_RModel(Reaction_Model):

    def __init__(self,
                 cw: ChemWorld = None,
                 rng: RandomState = None,
                 fmodel_start_conditions: Dict = None,
                 **kwargs):
        super(Default_RModel, self).__init__(**kwargs)
        self.cw = cw
        self.rng = rng

        if fmodel_start_conditions:
            self.__prepare_starting_molecule_set(self, **fmodel_start_conditions)
        else:
            self.starting_reactants = None

    def __prepare_starting_molecule_set(self,
                                        source_dir: os.PathLike = None,
                                        train_mode: bool = True,
                                        max_size: int = 6,
                                        min_reactions: int = 15,
                                        ):
        if not source_dir:
            # use the one from CW
            source_dir = self.cw.CP_DIR
        if os.path.exists(source_dir) and os.path.isdir(source_dir) and \
                os.path.exists(os.path.join(source_dir, 'starting_mols_train.csv')) and \
                os.path.exists(os.path.join(source_dir, 'starting_mols_test.csv')):
            # if the molecule files exist, use one of those
            if train_mode:
                self.starting_reactants = pd.read_csv(os.path.join(source_dir, 'starting_mols_train.csv'))
                return
            else:
                self.starting_reactants = pd.read_csv(os.path.join(source_dir, 'starting_mols_test.csv'))
                return
        else:
            # otherwise, generate the files and dump them
            ensure_dir_exists(source_dir)

            # identify the starting molecule candidates
            eligible = self.cw.reactants[
                (self.cw.reactants['n_heavy'] <= max_size) & (self.cw.reactants['n_templates'] >= min_reactions)]
            # now we have to separate into train and test set
            test_set = eligible['smiles'].sample(n=len(eligible)//2, replace=False, random_state=self.rng)
            train_set = eligible['smiles'][~eligible['smiles'].isin(test_set)]

            # save the datasets
            test_set.to_csv(os.path.join(source_dir, 'starting_mols_test.csv'))
            train_set.to_csv(os.path.join(source_dir, 'starting_mols_train.csv'))

            # call recursively to load from the csv file
            return self.__prepare_starting_molecule_set(source_dir, train_mode, max_size, min_reactions)

    def seed(self, rng: RandomState = None, seed=None):
        if rng:
            self.rng = rng
        elif seed:
            self.rng = np.random.RandomState(seed=seed)
        else:
            raise Exception("Failed to seed the reaction model, no seed or rng specified")

    def get_action_spec(self):
        return SpaceDict([Box(low=-3, high=3, shape=(self.cw.action_fp_len, )),
                          Discrete(self.cw.template_count)])

    def get_observation_spec(self):
        # since all the data is normalized, ideally, -3 to 3 should contain all the training set compounds available
        return SpaceDict([Box(low=-3, high=3, shape=(self.cw.observation_fp_len, )),
                          Box(low=-3, high=3, shape=(self.cw.template_count, ))])

    def forward_react(self, R1, template, R2):
        '''

        Parameters
        ----------
        R1
            reactant1, usually the compound that was there in the beginning of the step, SMILES
        template
            template_id selected
        R2
            reactant2, usually a molecule picked prior from the dataset, SMILES
        ed
            EnvData object

        Returns
        -------
        resultant molecule (major product), SMILES
        '''
        assert self.rng, "RNG not initializes"
        assert self.cw, "Chemical resources not loaded"
        m1 = Chem.MolFromSmiles(R1)
        rxn = self.cw.get_reaction_by_index(self.cw.get_reaction_index_from_template_index(template))

        if R2 == self.null_molecule:
            products = rxn.RunReactants((m1,))
        else:
            t_id = self.cw.get_template_order_from_template_index(template)
            m2 = Chem.MolFromSmiles(R2)
            products = None
            if t_id == 0:
                products = rxn.RunReactants((m1, m2))
            elif t_id == 1:
                products = rxn.RunReactants((m2, m1))
            else:
                raise InvalidAction("More than 2 reactants in reaction specified.")

        if len(products) == 0:
            raise ReactionFailed(AllChem.ReactionToSmarts(rxn), [R1, R2], template, rxn.Validate())

        prds = [elem for sublist in products for elem in sublist]
        smiles_products = list(set([SMol.standardize_light(p) for p in prds]))

        for prod in smiles_products:
            # check that its not an empty reagent as I believe that that
            # is what is breaking the score function occasionally
            if prod and prod != '':
                ret = ChemMolFromSmilesWrapper(prod)
                if ret:
                    return prod

        # if here, that means that we have failed to get any viable products from the reaction
        raise NoKekulizedProducts(smiles_products, AllChem.ReactionToSmarts(rxn), [R1, R2], template, rxn.Validate())

    def get_reactant(self, mol_vec, template):
        '''
        Here we:
            - get the conjugate template for the selected template
            - get all the compatible molecules from the reactant set
            - use kNN to select one nearest to it
            - return that reactants SMILEs

        A potentially very slow part. Since we got to find the nearest neighbors out of a lot of molecules.
        Parameters
        ----------
        mol_vec
            action vector
        template
            template integer

        Returns
        -------
        molecule
            SMILES of the molecule selected to be the second reagent
        '''
        assert self.rng, "RNG not initializes"
        assert self.cw, "Chemical resources not loaded"

        template_id = template
        all_templates = self.cw.get_templates_for_reaction_index(self.cw.get_reaction_index_from_template_index(template_id))
        other_templates = filter(lambda x: x != template_id, all_templates)

        if len(all_templates) == 1:
            # this means this is an intramolecular reaction
            return self.null_molecule
        elif len(all_templates) == 2:
            # just one other reactant
            other_template = next(other_templates)

            # do the KNN thing and pick the nearest reactant
            reactants_subset = self.cw.get_compatible_reactant_subset(other_template)
            react_fps = self.cw.action_fps_numpy(reactants_subset)

            neigh = NearestNeighbors(algorithm="brute", metric=self.cw.action_fp_dist_fn, n_jobs=6).fit(react_fps)
            # TODO: collect k nearest neighbors and softmax random over the distances to pick one
            ndist, n_ind = neigh.kneighbors(X=mol_vec.reshape(1, -1), n_neighbors=1, return_distance=True)
            selected_mol_index = reactants_subset[reactants_subset].index[n_ind.item()]  # get the index of the selected compound
            selected_mol = self.cw.get_reactant_smiles(selected_mol_index)
            #   (if the reactant df is not indexed evenly) (mostly fixed, but still watch out for this part)
            return selected_mol
        else:
            raise Exception("Up to binary reactions are supported ({} complementary templates found)."
                            .format(other_templates))

    def encode_observation(self, molecule):
        '''

        Parameters
        ----------
        molecule
            current molecule SMILES

        Returns
        -------
        molecule
            the molecule to encode
        '''
        assert self.rng, "RNG not initialized"
        assert self.cw, "Chemical resources not loaded"

        mol_vec = self.cw.observation_fp_fn(molecule)
        template_mask = self.cw.evaluate_reactivity_mask(molecule)

        return mol_vec, template_mask

    def verify_action(self, observation, avec, template, raise_errors=False):
        if observation[1][template] != 1:
            if raise_errors:
                raise InvalidAction("Reaction template {} cannot be applied to molecule '{}' (template mask is '{}').\n"
                                    .format(template, self.current_mol, self.current_obs[1]))
            else:
                return False
        return True

    def verify_state(self, observation, raise_errors=False):
        if np.sum(observation[1]) < 1:
            if raise_errors:
                raise ResetNeeded("The reaction model can no longer find a viable reaction template for the compound.\n"
                                  .format(self.current_mol, self.current_obs[1]))
            else:
                return False
        return True

    def init_state(self, max_size=6, min_reactions=15):
        if self.starting_reactants:
            return self.starting_reactants.sample(random_state=self.rng).item()
        else:
            # work in legacy mode if no starting reactants are provided
            eligible = self.cw.reactants[
                (self.cw.reactants['n_heavy'] <= max_size) & (self.cw.reactants['n_templates'] >= min_reactions)]
            return eligible['smiles'].sample(random_state=self.rng).item()

    def compute_diff(self, obs1, obs2):
        v1 = obs1[0]
        v2 = obs2[0]
        return self.cw.observation_fp_dist_fn(v1, v2)

    def _repr_str_(self):
        '''
        Produces a string representation of the chain of reactions.
        Returns
        -------

        '''
        assert self.record_history, "Recording history is not enabled, cannot visualise."

        # if history is enabled, there is always at least one molecule there
        accumulator = self.history[0]['new_molecule']

        for transdict in self.history[1:]:
            # find the reaction
            reaction = self.cw.get_reaction_index_from_template_index(transdict['reaction'])

            # find the other reactant and the product
            other_reactant = transdict['other_reactant']
            new_molecule = transdict['new_molecule']

            if other_reactant != self.null_molecule:
                accumulator = additive_template_R2.format(accumulator, other_reactant)

            accumulator = additive_template.format(accumulator, reaction, new_molecule)

        return accumulator

    def _repr_svg_(self):
        '''
        Produce an SVG representation (returned as string).
        SVG is chosen to avoid PIL and stuff like that in the dependencies.
        Returns
        -------
            string of the SVG.
        '''
        assert self.record_history, "Recording history is not enabled, cannot visualise."

        # if history is enabled, there is always at least one molecule there
        accumulator = self.history[0]['new_molecule']
        score = self.history[0]['new_mol_score']

        doc = ss.Document()
        main_layout = ss.HBoxLayout()
        doc.setLayout(main_layout)

        for transdict in self.history[1:]:
            # build the accumulator molecule image ()
            hlayout_mol = ss.VBoxLayout()
            main_layout.addLayout(hlayout_mol)

            hlayout_mol.addSVG(SmilesToSVG(accumulator, dims=(DEFAULT_MOL_WIDTH,
                                                              DEFAULT_MOL_HEIGHT-DEFAULT_SCORE_TEXT_HEIGHT
                                                              )))
            hlayout_mol.addSVG(swag_score.format(score))

            # now if we have a reaction, we gotta add that to the image
            reaction_caption = self.cw.get_reaction_str_repr(
                self.cw.get_reaction_index_from_template_index(transdict['reaction']))
            second_reactant = transdict['other_reactant']

            # draw those things
            hlayout_mol = ss.VBoxLayout()

            hlayout_mol.addSVG(SmilesToSVG(second_reactant,
                                           dims=(int(DEFAULT_MOL_WIDTH*0.6),
                                                 DEFAULT_MOL_HEIGHT-DEFAULT_RXN_TEXT_HEIGHT-DEFAULT_ARROW_HEIGHT,
                                                 )),
                               alignment=ss.AlignHCenter)
            hlayout_mol.addSVG(swag_arrow)
            hlayout_mol.addSVG(swag_rxn_text.format(reaction_caption))
            main_layout.addLayout(hlayout_mol, )

            # update for the next step
            accumulator = transdict['new_molecule']
            score = transdict['new_mol_score']

        # draw the last molecule in the chain
        hlayout_mol = ss.VBoxLayout()
        hlayout_mol.addSVG(SmilesToSVG(accumulator, dims=(DEFAULT_MOL_WIDTH,
                                                          DEFAULT_MOL_HEIGHT-DEFAULT_SCORE_TEXT_HEIGHT,
                                                          )))
        hlayout_mol.addSVG(swag_score.format(score))
        main_layout.addLayout(hlayout_mol)

        # finalize and return
        return doc.serialize()

    @property
    def null_template(self):
        return -1

    @property
    def null_molecule(self):
        return ''

    def suggest_action(self):
        # TODO: possibly make sampling more direct, with sampling the list of possible molecules directly.
        raise NotImplemented
        # a helper function mostly for debug purposes
        # molvec = self.action_space.sample()[0]
        # reaction_id = np.where(self.rmodel.get_current_state()[1])[0]
        # if reaction_id.sum() == 0:
        #     raise InvalidAction("No action to suggest, the environment is done.")
        # action = self.rng.choice(reaction_id, size=1).item()
        # # now that we have a reaction, we can compute eligible compounds and sample
        #
        # return tuple([molvec, action])


fmodel_dict = {
    'str_repr': Default_RModel,
}


def get_forward_model(type: str):
    return fmodel_dict[type]