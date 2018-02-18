# =========================
#       Configuration
# =========================

# ===========
#   General
# ===========
SEED = 2018             # Seed to reproduce experiments. Use `None` to run different experiments each time

# ===============================
#   Hindsight Experience Replay
# ===============================
HER = False             # Switch on to activate HER
HER_NEW_GOALS = 5       # Number of new goals for experience

# ============
#   Training
# ============
EPOCHS = 384            # Number of iterations to sample and train policies
EPISODES = 8            # Number of episodes sampled in each epoch
TRAINING_STEPS = 32     # Number of steps training policy on minibatch
BATCH_SIZE = 32         # Minibatch size for training
UPDATE_ACTOR = 8        # Update actor with critic after this amount of epochs

# ===========
#   Testing
# ===========
TESTING_EPISODES = 128  # Number of episodes used to test the success rate of the policy

# ==========
#   Buffer
# ==========
BUFFER_SIZE = 16384     # Size of the Buffer Experience
ERASE_FACTOR = .01      # Remove oldest `ERASE_FACTOR` percent of the buffer experience when it gets full

# ==============
#   Q Learning
# ==============

# This parameters were selected empirically without any fine tuning
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
