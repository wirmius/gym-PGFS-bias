import numpy as np

from typing import Callable
from abc import ABC, abstractmethod
from gym.error import InvalidAction

class Reaction_Model(ABC):
    '''
    An abstract class for all potential reaction models for the PGFS environments.

    The type format is the following:
    wmol : whole molecule data structure (eg smiles string or a mol object)
    ovec : obsesrvation vector (eg numpoy array)
    avec : action vector (eg a numpy array)
    temp : something uniquely identifying a reaction within this structure (eg an id in a pandas array)

    these 4 are expected to be ubiquitous within all the abstract methods.
    '''

    def __init__(self,
                 record_history = False,
                 **kwargs):
        self.current_mol = None
        self.current_obs = None

        self.record_history = record_history
        if self.record_history:
            self.history = []


    @abstractmethod
    def seed(self, rng=None, seed=None):
        pass

    @abstractmethod
    def init_state(self):
        pass

    @abstractmethod
    def forward_react(self, R1, template, R2):
        pass

    @abstractmethod
    def get_reactant(self, mol_vec, template):
        pass

    @abstractmethod
    def encode_observation(self, molecule):
        pass

    @abstractmethod
    def verify_action(self, observation, avec, template, raise_errors=False):
        pass

    @abstractmethod
    def verify_state(self, observation, raise_errors=False) -> bool:
        pass

    @abstractmethod
    def get_action_spec(self):
        pass

    @abstractmethod
    def get_observation_spec(self):
        pass

    @abstractmethod
    def seed(self, seed):
        pass

    @property
    @abstractmethod
    def null_molecule(self):
        pass

    @property
    @abstractmethod
    def null_template(self):
        pass

    def reset(self):
        self.current_mol = self.init_state()
        self.current_obs = self.encode_observation(self.current_mol)

        if self.record_history:
            self.history = [{'other_reactant': self.null_molecule, 'reaction': self.null_template,
                             'new_molecule': self.current_mol}]

        return self.current_obs

    def step(self, avec, template):
        ''' '''
        # verify that the reaction chosen is available for the given compound before proceeding
        if not self.verify_action(self.current_obs, avec, template):
            raise InvalidAction("Reaction template {} cannot be applied to molecule '{}' (current observation is {}).\n"
                                .format(template, self.current_mol, self.current_obs))

        # go as per the algorithm in the paper
        mol2 = self.get_reactant(avec, template)  # combines the GetValidReactants and kNN from [1]
        newmol = self.forward_react(self.current_mol, template, mol2)
        newobs = self.encode_observation(newmol)

        self.current_mol = newmol
        self.current_obs = newobs

        if self.record_history:
            self.history.append({'other_reactant': mol2, 'reaction': template, 'new_molecule': self.current_mol})

        return self.current_obs

    def get_current_state(self):
        '''

        Returns
        -------
        The observation of the current molecule
        '''
        return self.current_obs

    def get_current_molecule(self):
        '''

        Returns
        -------
        Current molecule in whatever format it is
        '''
        return self.current_mol

    @abstractmethod
    def suggest_action(self):
        pass


