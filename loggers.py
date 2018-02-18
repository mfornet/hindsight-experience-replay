import logging
from os.path import join
from sys import stdout

from config import *


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s| %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(stdout)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

    return logger


logger_her = setup_logger('logger_her', join(LOGS_PATH, 'logger_her.log'))
logger_her.disabled = False

logger_debug = setup_logger('logger_debug', join(LOGS_PATH, 'logger_debug.log'))
logger_debug.disabled = True

logger_episode = setup_logger('logger_episode', join(LOGS_PATH, 'logger_episode.log'))
logger_episode.disabled = True
