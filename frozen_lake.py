import gym
import numpy as np
import value_iteration, policy_service, policy_iteration, q_learning, stats_service, plotting_service
from gym.envs.registration import register
import sys
import time
from gym import envs


def run_value_iteration(env, gamma=0.9):

    number_trials = 100
    number_iterations_list = []
    trial_times = []

    for trial in range(number_trials):
        start_time = time.time()
        V, number_iterations = value_iteration.run(env, gamma=gamma)
        trial_times.append(time.time() - start_time)
        number_iterations_list.append(number_iterations)

    stats_service.print_convergence_stats(number_trials, number_iterations_list, trial_times)

    policy_service.print_V(V, env)

    optimal_policy = policy_service.get_optimal_policy_from_V_and_P(V, env.P, range(env.nA))

    policy_service.print_policy(optimal_policy, env)

    policy_success_rate, mean_number_steps_per_success_episode = policy_service.follow_policy(env, optimal_policy, number_episodes=number_trials)

    stats_service.print_success_stats(policy_success_rate, mean_number_steps_per_success_episode)
    return policy_success_rate


def run_policy_iteration(env, gamma=0.9):

    number_trials = 100
    number_iterations_list = []
    trial_times = []

    for trial in range(number_trials):
        start_time = time.time()
        optimal_policy, number_iterations = policy_iteration.run(env, gamma=gamma)
        trial_times.append(time.time() - start_time)
        number_iterations_list.append(number_iterations)

    stats_service.print_convergence_stats(number_trials, number_iterations_list, trial_times)

    policy_service.print_policy(optimal_policy, env )

    policy_success_rate, mean_number_steps_per_success_episode = policy_service.follow_policy(env, optimal_policy)

    stats_service.print_success_stats(policy_success_rate, mean_number_steps_per_success_episode)

    return policy_success_rate


def run_q_learning(env, determ_or_stoch, grid_size, gamma=0.9):

    number_trials = 100
    number_iterations_list = []
    trial_times = []
    frozen_lake_env_deterministic = get_deterministic_frozen_lake(grid_size)

    for trial in range(number_trials):

        start_time = time.time()
        if determ_or_stoch == 'd':
            Q, number_iterations = q_learning.run(frozen_lake_env_deterministic, gamma=gamma, max_iterations=250)
        else:
            Q, number_iterations = q_learning.run(env, gamma=gamma, max_iterations=2500)

        trial_times.append(time.time() - start_time)
        number_iterations_list.append(number_iterations)

    stats_service.print_convergence_stats(number_trials, number_iterations_list, trial_times)

    if determ_or_stoch == 'd':
        env = frozen_lake_env_deterministic

    policy = np.argmax(Q, axis=1)
    # policy = np.reshape(policy, env.desc.shape)
    policy_service.print_policy(policy, env)

    policy_success_rate, mean_number_steps_per_success_episode = policy_service.follow_policy(env, policy)

    stats_service.print_success_stats(policy_success_rate, mean_number_steps_per_success_episode)

    return policy_success_rate


def get_deterministic_frozen_lake(grid_size):
    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if 'FrozenLakeNotSlippery-v0' not in env_ids:
        # https://colab.research.google.com/drive/1u0saGlFdhlBqgX7q1lh4bTNr3kv1bRK_#scrollTo=kNxamkqSEfHB
        map_name = '8x8' if grid_size != '4' else '4x4'
        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': map_name, 'is_slippery': False},
            max_episode_steps=2000,
            reward_threshold=0.78,  # optimum = .8196
        )

    return gym.make('FrozenLakeNotSlippery-v0').env


def generate_gamma_plot(env, title_prefix, determ_or_stoch='d', grid_size='4'):
    gammas = np.arange(0, 1.001, 0.025)
    policy_success_rates = []
    for gamma in gammas:
        if title_prefix == 'VI':
            success_rate = run_value_iteration(env, gamma=gamma)
        elif title_prefix == 'PI':
            success_rate = run_policy_iteration(env, gamma=gamma)
        elif title_prefix == 'Q-Learn':
            success_rate = run_q_learning(env, determ_or_stoch, grid_size, gamma)
        else:
            success_rate = 0.0
        print(gamma, success_rate)
        policy_success_rates.append(success_rate)

    plotting_service.plot_line('{0} Policy Success Rate Per Gamma'.format(title_prefix), 'Gamma', 'Success Rate', gammas,
                               policy_success_rates)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Value Iteration: python frozen_lake.py 4 v')
        print('Policy Iteration: python frozen_lake.py 4 p')
        print('Q-Learning: python frozen_lake.py 4 q')
        exit()

    np.set_printoptions(precision=3)

    grid_size = sys.argv[1]
    if grid_size == '8':
        frozen_lake_env = gym.make('FrozenLake8x8-v0').env
    else:
        frozen_lake_env = gym.make('FrozenLake-v0').env

    function_to_run = sys.argv[2]
    if function_to_run == 'v':
        if len(sys.argv) > 3 and sys.argv[3] == 'gamma-plot':
            generate_gamma_plot(frozen_lake_env, 'VI')
        else:
            run_value_iteration(frozen_lake_env, gamma=0.9)
    elif function_to_run == 'p':
        if len(sys.argv) > 3 and sys.argv[3] == 'gamma-plot':
            generate_gamma_plot(frozen_lake_env, 'PI')
        else:
            run_policy_iteration(frozen_lake_env, gamma=0.9)
    elif function_to_run == 'q':

        deterministic_or_stochastic = sys.argv[3]
        if len(sys.argv) > 4 and sys.argv[4] == 'gamma-plot':
            generate_gamma_plot(frozen_lake_env, 'Q-Learn', deterministic_or_stochastic, grid_size)
        else:
            run_q_learning(frozen_lake_env, deterministic_or_stochastic, grid_size, gamma=0.9)
    else:
        print("Only v, p and q are valid command line options")





