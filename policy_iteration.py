import numpy as np
import policy_service


def run(env, max_iterations=10, gamma=0.9):

    # start with a random policy
    current_policy = np.random.choice(env.nA, env.nS)

    policy_did_not_improve_count = 0
    for iteration in range(max_iterations):

        V = evaluate_policy(current_policy, env, gamma=gamma)
        improved_policy = policy_service.get_optimal_policy_from_V_and_P(V, env.P, range(env.nA))

        if np.array_equal(current_policy, improved_policy):
            policy_did_not_improve_count += 1
        else:
            policy_did_not_improve_count = 0

        if policy_did_not_improve_count == 3:
            return improved_policy, iteration - 2

        current_policy = improved_policy

    print("Policy Iteration did NOT converge.")
    return current_policy, max_iterations


def evaluate_policy(policy, openai_gym_env, gamma=0.9, convergence_threshold=0.00000000001, max_iterations=1000):

    # follow the policy and calculate V( the states' values) for this policy
    P = openai_gym_env.P

    V = np.zeros(policy.shape[0])

    for iteration in range(max_iterations):

        max_converge_diff = 0
        V_t_minus_one = V.copy()
        V[:] = 0
        for state, action in enumerate(policy):
            for prob, s_prime, reward, is_s_prime_terminal in P[state][action]:

                if is_s_prime_terminal:
                    V[state] += prob * reward
                else:
                    V[state] += prob * (reward + gamma * V_t_minus_one[s_prime])

            max_converge_diff = max(max_converge_diff, abs(V_t_minus_one[state] - V[state]))

        if max_converge_diff < convergence_threshold:
            break

    return V


