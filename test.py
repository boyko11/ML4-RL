import numpy as np
import gym
import policy_service
import policy_iteration
from gym.envs.registration import register
from collections import deque

# test_array = np.array([1, 2, 3, 4, 5])
#
# selections = []
# for i in range(1000):
#     selections.append(np.random.choice(test_array, 1)[0])
#
# unique, counts = np.unique(np.array(selections), return_counts=True)
# print(unique, counts)

# Q = np.random.uniform(0, 10, 16)
# Q = np.reshape(Q, (4, 4))
#
# print(Q)
# print(np.max(Q))
#
# exit()

# frozen_lake_env = gym.make('FrozenLake-v0').env

print(np.arange(0, 1.001, 0.025))
exit()

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=2000,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0')

exit()

# print(frozen_lake_env.P[0].keys())
#
# print(len(frozen_lake_env.P))
#
# random_actions = np.random.choice(tuple(frozen_lake_env.P[0].keys()), len(frozen_lake_env.P))
#
# print(random_actions)
#
# random_actions = np.random.choice(frozen_lake_env.nA, frozen_lake_env.nS)
#
# print(random_actions)

def policy_evaluation(pi, P, gamma=0.9, theta=1e-10):
    V = np.zeros(len(pi))

    while True:
        max_delta = 0
        old_V = V.copy()

        for s in range(len(P)):
            V[s] = 0
            for prob, new_state, reward, done in P[s][pi[s]]:
                if done:
                    value = reward
                else:
                    value = reward + gamma * old_V[new_state]
                V[s] += prob * value
            max_delta = max(max_delta, abs(old_V[s] - V[s]))
        if max_delta < theta:
            break
    return V.copy()

def policy_improvement(pi, V, P, gamma=0.9):

    pi_copy = pi.copy()
    for s in range(len(V)):
        Qs = np.zeros(len(P[0]), dtype=np.float64)
        for a in range(len(P[s])):
            for prob, new_state, reward, done in P[s][a]:
                if done:
                    value = reward
                else:
                    value = reward + gamma * V[new_state]
                Qs[a] += prob * value
        pi_copy[s] = np.argmax(Qs)
    return pi_copy

pi = np.array([2, 0, 1, 3, 0, 0, 2, 0, 3, 1, 3, 0, 0, 2, 1, 0])

print("Using Miguel Morales' code as benchmark Start")
print("Policy to be evaluated: ")
policy_service.print_policy(pi, frozen_lake_env)
V = policy_evaluation(pi, frozen_lake_env.P)
policy_service.print_V(V, frozen_lake_env)
improved_pi = policy_improvement(pi, V, frozen_lake_env.P)
print("Improved policy: ")
policy_service.print_policy(improved_pi, frozen_lake_env)

print("Using Miguel Morales' code as benchmark END")

print('-------------------------------------------')

print("This implementation Start")
print("Policy to be evaluated: ")
policy_service.print_policy(pi, frozen_lake_env)
V = policy_iteration.evaluate_policy(pi, frozen_lake_env)
policy_service.print_V(V, frozen_lake_env)
improved_pi = policy_service.get_optimal_policy_from_V_and_P(V, frozen_lake_env.P, range(frozen_lake_env.nA))
print("Improved policy: ")
policy_service.print_policy(improved_pi, frozen_lake_env)


print("This implementation End")