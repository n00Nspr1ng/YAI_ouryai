import numpy as np


class Demo(object):

    def __init__(self, observations, random_seed=None, num_reset_attempts = None):
        self._observations = observations
        self.random_seed = random_seed
<<<<<<< HEAD
        self.num_reset_attempts = num_reset_attempts
=======
        self.variation_number = 0
>>>>>>> 1690e9321fd9f1e9a5680127ad53c01ed028db40

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)
