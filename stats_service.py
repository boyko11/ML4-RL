import numpy as np


def print_convergence_stats(number_trials, number_iterations_list, trial_times):

    print("Average number of iterations it took to converge over {0} trials: {1}".format(number_trials,
                                                                             np.mean(np.array(number_iterations_list))))
    print("Average convergence time over {0} trials: {1}".format(number_trials, np.mean(np.array(trial_times))))


def print_success_stats(policy_success_rate, mean_number_steps_per_success_episode):

    print("Optimal policy success rate: {0}".format(policy_success_rate))
    print("Avg number of steps to the Goal: {0}".format(mean_number_steps_per_success_episode))