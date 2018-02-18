# Hindsight Experience Replay

## Environment

## Algorithm

    N := Vector size of status
    B := Buffer experience replay
    Initialize `B` with random experience

    Initialize Critic and Actor Neural Networks

    For epoch = 1..EPOCHS:
        For episodes = 1..EPISODES:
            g := Sample random goal status
            st_0 := Sample initial status
            s_0 := (st_0, g) # Initial state

            For t = 0..N-1:
                a_t = Sample action using epsilon greedy policy according to critic policy
                s_{t+1}, r_t = Observe state after action a_t on state s_t

            For t = 0..N-1:
                Add to B experience (s_t, a_t, r_t, s_{t+1})

                Sample status positions in current episode st_r:
                    Add to B this experience with different goal and updated reward

        For t=1..TRAIN:
            MB := Sample a minibatch from B
            Perform an optimization step on Critic

        If K epoch has passed since actor update:
            Update actor weights with critic weights

## Notes:

    + Status are vector s.t. v \in {0,1}^N
    + N is the size of such vectors
    + State is a tuple of current status, goal status

# TODO: Visualize learning curve
# TODO: Document the code
# TODO: Generate the target graphics