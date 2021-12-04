import numpy as np

import gym
from gym.spaces import Space, Tuple as SpaceDict, MultiBinary, Discrete
from gym.error import ResetNeeded, InvalidAction
from gym_PGFS.forward_model import Reaction_Model
from gym_PGFS.function_set_basic import Default_RModel
from gym_PGFS.scoring_functions import get_scoring_function

from typing import Callable, Type, Union


class PGFS_env(gym.Env):
    """
    PGFS environment as described in Gottipati et al. (2020)

    Some notation:
        -   observations space:
            *   (1) the available reaction templates mask and (2) the current molecule;
        -   action space:
            *   (1) the reaction template and (2) the molecule to react with; (if possible can be made modular to quick
                  swap it with REACTOR style);
        -   info:   can drop a dict of multiple things like
        -   rewards:

    Important remarks:
        * the environment has to store the full smiles string of the current R(1) in order to avoid losses and mess from
         uncompressing the fingerprints (the agent doesnt need to know of those things);
        
    """
    def __init__(self,
                 scoring_fn: Union[Callable, str],  # scoring function
                 scoring_transform: str = 'none',
                 rng_seed=778821916,
                 give_info: bool = False,
                 max_steps: int = 10,
                 render: bool = False,
                 fmodel: Type[Reaction_Model] = Default_RModel,
                 **kwargs):
        '''
        There are several functions that can be switched out in the algorithm proposed by Gottipati et al.
        They and some other parameters can be specified during the initialization.

        Parameters
        ----------
        rng_seed :
            Seed for the rng
        give_info :
            whether to return info during steps
        kwargs :
            forwarded to the reaction model
        '''
        super(PGFS_env, self).__init__()

        self.give_info = give_info
        self.max_steps = max_steps
        self.enable_render = render

        # assign gym.environment variables

        self.rng = None
        self.current_mol = None
        self.current_obs = None
        self.current_step = 0
        self.kwargs = kwargs

        self.rmodel = fmodel(record_history=self.enable_render, **kwargs)

        if isinstance(scoring_fn, str):
            self.scoring_fn = get_scoring_function(scoring_fn, scoring_transform=scoring_transform)
        else:
            self.scoring_fn = scoring_fn
        self.observation_space = self.rmodel.get_observation_spec()
        self.action_space = self.rmodel.get_action_spec()

        self.seed(rng_seed)

    def seed(self, seed):
        ''' Seed all the random elements and initialize the local rng. '''
        self.rng = np.random.RandomState(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.rmodel.seed(rng=self.rng)
        self.kwargs['rng'] = self.rng  # add to the kwargs

    def reset(self):
        '''
        Clear the current environment state. Produce the initial state with an RNG in mind
        '''
        self.current_step = 0
        state = self.rmodel.reset()

        if self.enable_render:
            # if rendering is enabled, provide the scores for the molecules
            self.rmodel.history[-1]['new_mol_score'] = self.scoring_fn(self.rmodel.get_current_molecule())

        return state

    def step(self, action):

        done = False
        if self.current_step >= self.max_steps:
            raise ResetNeeded
        elif self.current_step == self.max_steps - 1:
            done = True

        # make sure that the action is valid
        self.rmodel.verify_action(self.rmodel.get_current_state(), *action, raise_errors=True)

        # increment the step
        self.current_step += 1

        # step the reaction model
        newstate = self.rmodel.step(*action)

        # check if the environment is done
        done = done or not self.rmodel.verify_state(newstate, raise_errors=False)

        # compute reward
        # reward = self.scoring_function(newstate, **self.kwargs)
        reward = self.scoring_fn(self.rmodel.get_current_molecule())

        info = {}
        if self.give_info:
            info["new_molecule"] = self.rmodel.get_current_molecule()
            info["old_mol_template"] = action[1]
            # add more if needed

        if self.enable_render:
            # if rendering is enabled, provide the molecule history with the score of the given molecule
            self.rmodel.history[-1]['new_mol_score'] = reward

        return newstate, reward, done, info

    def suggest_action(self):
        # a helper function mostly for debug purposes
        molvec = self.action_space.sample()[0]
        reaction_id = np.where(self.rmodel.get_current_state()[1])[0]
        if reaction_id.sum() == 0:
            raise InvalidAction("No action to suggest, the environment is done.")
        action = self.rng.choice(reaction_id, size=1).item()
        return tuple([molvec, action])

    def render(self, mode='human'):
        if self.enable_render:
            return self.rmodel._repr_svg_()
        else:
            return "environment rendering is disabled"

"""
TODO:
    top (essential function):
        +   implement the core skeleton;
        +   implement reaction mechanism availability prediction (so fused with vectorisation in encode_observation());
        +   implement reactant kNN (or other approach) prediction as per paper;
        +   implement a scoring function;
        +   implement an observation encoder;
        +   devise a molecule gym.Space
        
    medium (functionality by the paper and flexibility):
        +   make things modular (including the skeleton functions, the molecule representation, perhaps?)

    would be cool to (possible future features that are, as far as I know, have not been proposed in the paper):
        -   make sure we have advanced reaction features such as:
            * softmax product selection (see if you can extract the fractions of a specific product of a reaction)
        -   penalize larger molecules during the kNN stage???
        
        
    low (nice to haves):
        ~   __str__, __enter__ and __exit__ functions
        +   nice rendering of molecules
"""


"""
References:
[1] : Gottipati et al. 2019

"""
