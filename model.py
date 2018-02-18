from os.path import exists

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Concatenate, Dot
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from loggers import logger_her
from utils import *


class QModel:
    def __init__(self, n, her):
        self.n = n
        self.her = her

        self.checkpoint = ModelCheckpoint(model_path(self.n, self.her))
        self.build()

    def build(self):
        n = self.n

        status = Input(shape=(n,))
        goal = Input(shape=(n,))
        action = Input(shape=(n,))

        input_layer = Concatenate()([status, goal])
        hidden = Dense(256, activation='relu')(input_layer)

        action_value = Dense(n)(hidden)

        current_value = Dot(-1)([action_value, action])  # Value of the current action

        # Models with shared weights but different I\O
        # model_a is for training
        # model_b if for predicting output

        self.model_a = Model(inputs=[status, goal, action],
                             outputs=[current_value])

        self.model_b = Model(inputs=[status, goal],
                             outputs=[action_value])

        opt = Adam()
        self.model_a.compile(optimizer=opt, loss='mse')

        # This is necessary but useless since this model
        # will never be fitted.
        self.model_b.compile(optimizer=opt, loss='mse')

    def action_value(self, states):
        status = np.array([status for status, goal in states])
        goal = np.array([goal for status, goal in states])
        action_value = self.model_b.predict([status, goal])
        return action_value

    def best_action_value(self, states):
        status = np.array([status for status, goal in states])
        goal = np.array([goal for status, goal in states])
        action_value = self.model_b.predict([status, goal])

        best_value = action_value.max(1)
        best_action = action_value.argmax(1)

        return best_value, best_action

    def select_action(self, state, epsilon_greedy=None):
        if epsilon_greedy is not None and np.random.random() < epsilon_greedy:  # Sample random action
            action = np.random.randint(state.size)
        else:  # Sample greedy action
            _, _action = self.best_action_value([state])
            action = _action[0]

        return action

    def train(self, status, goal, action, target, *args, **kwargs):
        one_hot_action = to_categorical(action, self.n)
        self.model_a.fit([status, goal, one_hot_action], [target], callbacks=[self.checkpoint], *args, **kwargs)

    def update(self, critic):
        self.model_a.set_weights(critic.model_a.get_weights())

    def load(self):
        path = model_path(self.n, self.her)
        if exists(path):
            try:
                self.model_a.load_weights(path)
                logger_her.info("Model loaded...")
            except OSError:
                pass