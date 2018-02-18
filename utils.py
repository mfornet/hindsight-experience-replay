from os.path import join

from config import *


def model_path(size):
    return join(MODEL_PATH, 'weights-{}-{}.h5'.format(
        size, "her" if HER else "noher"))
