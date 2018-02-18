import numpy as np

from config import ERASE_FACTOR
from loggers import logger_her


class Buffer:
    """
        Experience replay buffer
    """
    factor = ERASE_FACTOR

    def __init__(self, size):
        self.max_size = size
        self.data = []

    @property
    def size(self):
        return len(self.data)

    def add(self, experience):
        self.data.extend(experience)
        if len(self.data) >= self.max_size:
            self.data = self.data[int(Buffer.factor * self.max_size):]

    def sample(self, size):
        replace_mode = size > len(self.data)
        index = np.random.choice(self.size, size=size, replace=replace_mode)
        return [self.data[idx] for idx in index]

    def log_stats(self):
        reward_count = np.zeros(2)
        for (s, a, r, sn) in self.data:
            reward_count[-r] += 1
        reward_count /= reward_count.sum()

        logger_her.info("0/-1 reward: {}/{}".format(reward_count[0], reward_count[1]))
        logger_her.info("Stored experience: {}".format(self.size))