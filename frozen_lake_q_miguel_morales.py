import numpy as np
import gym
from tqdm import tqdm
import policy_service, stats_service
import sys, time

#https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_03/chapter-03.ipynb

def q_learning(env,
               gamma=0.9,
               initial_alpha=0.5,
               alpha_decay_rate=1e-3,
               min_alpha=0.001,
               initial_epsilon=1.,
               epsilon_decay_rate=1e-5,
               min_epsilon=0.0,
               n_episodes=20000):
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    Q_track = np.zeros((n_episodes, nS, nA))
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(nA)

    for t in tqdm(range(n_episodes)):
        alpha = max(initial_alpha * np.exp(-alpha_decay_rate * t), min_alpha)
        epsilon = max(initial_epsilon * np.exp(-epsilon_decay_rate * t), min_epsilon)
        state, done = env.reset(), False
        while not done:
            action = select_action(state, Q, epsilon)
            new_state, reward, done, _ = env.step(action)
            if done:
                Q[new_state] = 0.
                Q_est = reward
            else:
                Q_est = reward + gamma * Q[new_state].max()
            Q[state][action] = Q[state][action] + alpha * (Q_est - Q[state][action])
            state = new_state
        Q_track[t] = Q

    pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
    V = np.max(Q, axis=1)
    return Q, V, pi, Q_track



if __name__ == '__main__':

    if len(sys.argv) < 2:

        print('Q-Learning: python frozen_lake_q_miguel_morales.py 4')
        exit()

    np.set_printoptions(precision=3)

    grid_size = sys.argv[1]
    if grid_size == '8':
        env = gym.make('FrozenLake8x8-v0')
    else:
        env = gym.make('FrozenLake-v0')

    number_trials = 1
    number_iterations_list = []
    trial_times = []
    episodes = 25000
    for trial in range(number_trials):
        start_time = time.time()
        Q_best, V_best, pi_best, Q_track = q_learning(env=env, n_episodes=episodes)
        trial_times.append(time.time() - start_time)
        number_iterations_list.append(episodes)



    print(Q_best)
    print(V_best)
    optimal_policy = np.zeros(V_best.shape[0], np.int64)
    for s, a in pi_best.items():
        optimal_policy[s] = a

    stats_service.print_convergence_stats(number_trials, number_iterations_list, trial_times)

    policy_service.print_policy(optimal_policy, env.env )

    policy_success_rate, mean_number_steps_per_success_episode = policy_service.follow_policy(env.env, optimal_policy)

    stats_service.print_success_stats(policy_success_rate, mean_number_steps_per_success_episode)