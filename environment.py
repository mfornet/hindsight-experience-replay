import collections
import copy

import numpy as np


class State(collections.namedtuple("_State", ("status", "goal"))):
    @property
    def size(self):
        return self.status.shape[0]

    @property
    def is_final(self):
        return (self.status == self.goal).all()

    @classmethod
    def sample_status(cls, n):
        return np.random.randint(2, size=n)

    def step(self, action):
        assert 0 <= action < self.size
        n_state = copy.deepcopy(self)
        n_state.status[action] ^= 1
        reward = 0 if n_state.is_final else -1
        return n_state, reward

    def __str__(self):
        return '\n'.join(str(x) for x in self)
