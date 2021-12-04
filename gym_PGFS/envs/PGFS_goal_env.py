import numpy as np

import gym
from gym.spaces import Space, Tuple as SpaceDict, MultiBinary, Discrete, Dict as ActualSpaceDict
from gym.error import ResetNeeded, InvalidAction

from typing import Callable

from gym_PGFS.function_set_basic import Default_RModel


class PGFS_Goal_env(gym.GoalEnv):
    def __init__(self,
                 rng_seed=+420778821916,
                 goal = 'c1ccccc1',
                 max_steps=10,
                 enable_render=False,
                 **kwargs):
        super(PGFS_Goal_env, self).__init__()

        self.max_steps = max_steps
        self.enable_render = enable_render

        self.rng = None
        self.current_mol = None
        self.current_obs = None
        self.current_step = 0
        self.kwargs = kwargs

        self.rmodel = Default_RModel(**kwargs)
        self.observation_space = ActualSpaceDict({'observation': self.rmodel.get_observation_spec(),
                                                  'desired_goal': self.rmodel.get_observation_spec(),
                                                  'achieved_goal': self.rmodel.get_observation_spec()})
        self.action_space = self.rmodel.get_action_spec()

        if enable_render:
            self.mol_history = None

        self.seed(rng_seed)

        # set the environments goal
        self.goal = goal
        self.goal_obs = self.rmodel.encode_observation(goal)

        #
        self.done = True

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.rmodel.compute_diff(achieved_goal, desired_goal)

    def seed(self, seed):
        ''' Seed all the random elements and initialize the local rng. '''
        self.rng = np.random.default_rng(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.rmodel.seed(rng=self.rng)
        self.kwargs['rng'] = self.rng  # add to the kwargs

    def reset(self):
        '''
        Clear the current environment state. Produce the initial state with an RNG in mind
        '''
        super(PGFS_Goal_env, self).reset()
        self.done = False
        self.current_step = 0
        if self.enable_render:
            self.mol_history = []
        return {'observation': self.rmodel.reset(), 'achieved_goal': self.rmodel.reset(), 'desired_goal': self.goal_obs}

    def step(self, action):
        info = dict()
        if self.done:
            raise ResetNeeded
        elif self.current_step > self.max_steps:
            raise ResetNeeded
        elif self.current_step == self.max_steps:
            self.done = True

        # make sure that the action is valid
        self.rmodel.verify_action(self.rmodel.get_current_state(), *action, raise_errors=True)

        # increment the step
        self.current_step += 1

        # step the reaction model
        info["old_molecule"] = self.rmodel.get_current_molecule()
        newstate = self.rmodel.step(*action)
        info["new_molecule"] = self.rmodel.get_current_molecule()
        info["old_mol_template"] = action[1]

        # check if the environment is done
        done = self.rmodel.verify_state(newstate, raise_errors=False)

        # compute reward
        reward = self.compute_reward(newstate, self.goal_obs, info)
        if np.isclose(reward, 0):
            self.done = True

        if self.enable_render:
            self.mol_history.append(info)

        return {'observation': newstate, 'achieved_goal': newstate, 'desired_goal': self.goal_obs}, reward, self.done, info

    def suggest_action(self):
        # a helper function mostly for debug purposes
        molvec = self.action_space.sample()[0]
        reaction_id = np.where(self.rmodel.get_current_state()[1])[0]
        if reaction_id.sum() == 0:
            raise InvalidAction("No action to suggest, the environment is done.")
        action = self.rng.choice(reaction_id, size=1).item()
        return tuple([molvec, action])

    def render(self, mode='human'):
        pass