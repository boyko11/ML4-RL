import numpy as np


def run(env, gamma=0.9, max_iterations=500):
    np.set_printoptions(suppress=True)

    Q = np.random.uniform(0, 0, env.nS * env.nA)
    Q = np.reshape(Q, (env.nS, env.nA))

    state_action_pair_visits = np.full((env.nS, env.nA), 1, dtype=np.int64)

    flatten_desc = env.desc.flatten()
    start_index = np.where(flatten_desc == b'S')
    frozen_indices = np.where(flatten_desc == b'F')

    all_possible_start_states = np.append(start_index, frozen_indices)

    for iteration in range(max_iterations):
        env.reset()
        # pick a random starting state
        current_state = np.random.choice(all_possible_start_states, 1)[0]
        env.s = current_state
        done = False

        while not done:
            action = np.random.randint(env.nA)
            new_state, reward, done, info = env.step(action)
            alpha = 1.0 / float(state_action_pair_visits[current_state, action])
            state_action_pair_visits[current_state, action] += 1
            Q[current_state, action] = (1 - alpha) * Q[current_state, action] + alpha * (
                        reward + gamma * np.max(Q[new_state, :]))
            current_state = new_state

        # Q_diff = np.abs(Q - Q_episode_start)
        # max_diff = np.max(Q_diff)
        # if max_diff <= convergence_threshold:
        #     print("Converged after {0} iterations(episodes)".format(iteration + 1))
        #     print(Q)
        #     return Q

    # print(state_action_pair_visits)
    # print(Q)
    return Q, max_iterations