import numpy as np

from buffer import Buffer
from environment import State
from loggers import *
from model import QModel

# Seed random generator to reproduce experiments
np.random.seed(SEED)


def _sample(n, k):
    """ Select k number out of n without replacement unless k is greater than n
    """
    if k > n:
        return np.random.choice(n, k, replace=True)
    else:
        return np.random.choice(n, k, replace=False)


def sample_episode(actor, state, epsilon_greedy, verbose=0):
    success = False
    experience = []
    eps = EPSILON if epsilon_greedy else None

    logger_episode.info("*** Begin episode ***")
    logger_episode.info("Status:{} Goal:{}".format(state.status, state.goal))

    for step in range(state.size):
        action = actor.select_action(state, eps)
        n_state, reward = state.step(action)

        logger_episode.info("Action:{} Reward:{}".format(action, reward))
        logger_episode.info("Status:{} Goal:{}".format(state.status, state.goal))

        experience.append((state, action, reward, n_state))
        if n_state.is_final:
            success = True
        state = n_state

    logger_episode.info("Success: {}".format(success))

    return success, experience


def evaluate_actor(actor, episodes_count=TESTING_EPISODES, verbose=0, pause=0):
    success_counter = 0

    for episode_ev in range(episodes_count):
        start = State.sample_status(actor.n)
        goal = State.sample_status(actor.n)
        success, _ = sample_episode(actor, State(start, goal), epsilon_greedy=False, verbose=verbose)
        success_counter += int(success)

        if pause: input("Press <Enter> to continue...")

    logger_her.info("Success/Total {}/{}".format(success_counter, episodes_count))
    logger_her.info("Success rate: {}".format(success_counter / episodes_count))

    return success_counter / episodes_count


def train(critic, actor, minibatch):
    status = np.array([state.status for (state, action, reward, n_state) in minibatch])
    goal = np.array([state.goal for (state, action, reward, n_state) in minibatch])
    action = np.array([action for (state, action, reward, n_state) in minibatch])
    target = np.zeros(len(minibatch))

    states = [state for (state, action, reward, n_state) in minibatch]
    n_states = [n_state for (state, action, reward, n_state) in minibatch]

    value = critic.action_value(states)
    n_value, _ = actor.best_action_value(n_states)

    for idx, (_, action_, reward_, _) in enumerate(minibatch):
        Q_s_a = value[idx][action_]
        n_Q_s_a = 0 if reward_ == 0 else (reward_ + DISCOUNT * n_value[idx])
        target[idx] = Q_s_a + ALPHA * (n_Q_s_a - Q_s_a)

    critic.train(status, goal, action, target, verbose=TRAIN_VERBOSE)


def loop(n):
    logger_her.info("***************************")
    logger_her.info("**** Bit flipping game ****")
    logger_her.info("***************************")

    logger_her.info("Start main loop with size {}".format(n))
    logger_her.info("HER STATUS: {}".format(HER))

    actor = QModel(n, HER)
    critic = QModel(n, HER)

    if not TRAIN_FROM_SCRATCH:
        actor.load()
        critic.load()
    else:
        logger_her.info("Training QNetworks from scratch")

    re_buffer = Buffer(BUFFER_SIZE)

    for epoch in range(EPOCHS):
        logger_her.info("Start epoch {}".format(epoch + 1))

        for episode_idx in range(EPISODES):
            goal = State.sample_status(n)
            start = State.sample_status(n)
            state = State(start, goal)

            _, episode = sample_episode(actor, state, epsilon_greedy=True)
            re_buffer.add(episode)

            if HER:
                new_experience = []
                for s, a, r, sn in episode:
                    for t in _sample(n, HER_NEW_GOALS):
                        _g = episode[t][-1].status
                        _sn = State(sn.status.copy(), _g.copy())

                        exp = (State(s.status.copy(), _g.copy()),
                               a,
                               0 if _sn.is_final else -1,
                               _sn)

                        new_experience.append(exp)

                re_buffer.add(new_experience)

        for training_step in range(TRAINING_STEPS):
            minibatch = re_buffer.sample(BATCH_SIZE)
            train(critic, actor, minibatch)

        if (epoch + 1) % UPDATE_ACTOR == 0:
            actor.update(critic)
            success_rate = evaluate_actor(actor)

            re_buffer.log_stats()

            if success_rate >= 1. - 1e-9:
                logger_her.info("Learned policy (QAction-Value) for {} bits in {} epochs".format(
                    n, epoch + 1
                ))
                break


if __name__ == '__main__':
    loop(11)
