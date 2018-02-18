# =========================
#       Configuration
# =========================

# ===========
#   General
# ===========
SEED = 2018  # Seed to reproduce experiments. Use `None` to run different experiments each time

# ===============================
#   Hindsight Experience Replay
# ===============================
HER = True  # Switch on to activate HER
HER_NEW_GOALS = 5  # Number of new goals for experience

# ============
#   Training
# ============
EPOCHS = 384
EPISODES = 8
TRAINING_STEPS = 32
BATCH_SIZE = 32  # Minibatch size for training
UPDATE_ACTOR = 8

# ===========
#   Testing
# ===========
TESTING_EPISODES = 128

# ==========
#   Buffer
# ==========
BUFFER_SIZE = 16384
ERASE_FACTOR = .01

# ==============
#   Q Learning
# ==============
DISCOUNT = .9
EPSILON = .1
ALPHA = .7

# ===========
#   Logging
# ===========
LOGS_PATH = 'logs'
TRAIN_VERBOSE = 0

# ===============
#   Checkpoints
# ===============
TRAIN_FROM_SCRATCH = False
MODEL_PATH = 'models'
