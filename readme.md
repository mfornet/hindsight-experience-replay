# Hindsight Experience Replay

Implementation of standard off-policy method in the bit-flipping environment augmented with HER.

The intention is trying to reproduce the [original paper](https://arxiv.org/abs/1707.01495) results. Details of the environment and the technique can be found there.

Different models were trained for each bit count. The success rate is the percent of episodes where the algorithms makes it from the start state to the goal. 

![HER results on the Bit Flipping Environment](https://github.com/mfornet/hindsight-experience-replay/raw/master/src/common/images/her.png "Bit-flipping results with and without her.")

The policy is unable to succeed for larger values of `N` because it doesn't have any sample with reward different from `-1`. This problem is fixed with HER improvement.

## Algorithm

    N := Number of bits
    B := Buffer experience replay

    Initialize Critic and Actor Neural Networks

    For epoch = 1..EPOCHS:
        For episodes = 1..EPISODES:
            g := Sample random goal status
            st_0 := Sample initial status
            s_0 := (st_0, g) # Initial state

            For t = 0..N-1:
                a_t = Sample action using epsilon greedy policy according to actor policy
                s_{t+1}, r_t = Observe state and reward after action a_t on state s_t

            For t = 0..N-1:
                Add to B experience (s_t, a_t, r_t, s_{t+1})

                Sample status positions in current episode st_r:
                    Add to B this experience with different goal and updated reward

        For t=1..TRAIN:
            MB := Sample a minibatch from B
            Perform an optimization step on critic policy using MB

        If K epoch has passed since last actor update:
            Update actor weights with critic weights
