import numpy as np


def run(openai_gym_env, gamma=0.9, convergence_threshold=0.00000000001, max_iterations=1000):

    P = openai_gym_env.P

    # actions[0] - left, actions[1] - down, actions[2] - right, actions[3] - up

    # Structure of P
    # 0: {
    #     1: [(0.3333333333333333, 0, 0.0, False),
    #         (0.3333333333333333, 4, 0.0, False),
    #         (0.3333333333333333, 1, 0.0, False)
    #         ],
    # If I am in state 0, take action 1

    # Value iteration V(s)

    # Start with arbitrary utilities
    V = np.random.random(openai_gym_env.nS)

    # update utilities based on neighbors
    # V(s)_t_plus_1 = r(s) + gamma * max_a( sigma_s_prime(T(s, a, s_prime) * V(s_prime))_t )

    # for each possible state
    # for each possible action
    # for each of the possible states the action can take us to

    for iteration in range(max_iterations):
        max_converge_diff = 0
        for state in range(openai_gym_env.nS):

            action_values = np.zeros(openai_gym_env.nA)
            for action_in_state in range(openai_gym_env.nA):

                for prob, s_prime, reward, is_s_prime_terminal in P[state][action_in_state]:

                    if is_s_prime_terminal:
                        action_values[action_in_state] += prob * reward

                    else:
                        action_values[action_in_state] += prob * (reward + gamma * V[s_prime])

            orig_v = V[state]
            V[state] = np.max(action_values)
            # print(state, action_values, V[state], (V[state] - orig_v))
            # print('---------------------------------------------------')
            max_converge_diff = max(max_converge_diff, abs(orig_v - V[state]))

        if max_converge_diff < convergence_threshold:
            # print('Took {0} iterations to converge'.format(iteration + 1))
            break

    return V, iteration + 1