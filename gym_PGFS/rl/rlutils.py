import numpy as np
import os
from collections import namedtuple
from typing import Union
import pickle

# s{R, T_mask}, a{T, a}, r, s{R_new, T_mask_new}
Transition = namedtuple('Transition', 'R_old T_mask a T reward R_new T_mask_new')

class Episode(object):
    def __init__(self):
        self.transitions = []

    def add(self, Transition):
        self.transitions.append(Transition)

    def compute_return(self, gamma):
        G = 0
        for t in self.transitions:
            G += t.reward*gamma
        return G

    def serialise_for_replay(self):
        pass

class ReplayBuffer(object):
    def __init__(self,
                 max_size=int(1e5),
                 random_state=np.random.RandomState
                 ):
        self.max_size = max_size
        self.counter = 0
        self.current_index = 0
        # dictionary lookup is faster for large buffers
        self.storage = dict()
        self.rs = random_state

    def __len__(self):
        content = min(self.counter, self.max_size)
        return content

    def add(self, transition):
        self.current_index = self.current_index % self.max_size
        self.storage[self.current_index] = transition
        self.current_index += 1
        self.counter += 1

    def sample(self,
               batch_size):
        indices = self.rs.randint(0, self.__len__(), size=batch_size).tolist()

        # now gather things into a batch
        ls_trans = [self.storage[i] for i in indices]

        # now convert the batch into a transition object ( for convenience )
        retval = Transition(
            *[
                np.concatenate(
                    [np.array(tr[i])[np.newaxis, :]
                     if not np.isscalar(tr[i])
                     else np.array(tr[i])[np.newaxis, np.newaxis]
                     for tr in ls_trans], axis=0)
                for i in range(len(ls_trans[0]))
            ])

        return retval
