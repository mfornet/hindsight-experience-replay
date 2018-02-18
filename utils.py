from os.path import join

from config import *


def model_path(size, her):
    return join(MODEL_PATH, 'weights-{}-{}.h5'.format(
        size, "her" if her else "noher"))
