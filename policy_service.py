import numpy as np


def get_optimal_policy_from_V_and_P(V, P, actions):

    pie_star = np.full(V.shape[0], -1, np.int)
    for state in range(V.shape[0]):
        action_values = np.zeros(len(actions))
        for action in actions:
            for prob, result_state, reward, is_result_state_terminal in P[state][action]:
                action_values[action] += prob * (reward if is_result_state_terminal else V[result_state])

        pie_star[state] = np.argmax(action_values)

    return pie_star


def follow_policy(env, policy, number_episodes=10):

    policy_copy = policy.copy()
    policy_copy = policy_copy.flatten()

    success_count = 0
    number_steps_list = []
    for episode in range(number_episodes):
        current_state = env.reset()
        done = False
        # frozen_lake_env.render()
        # input("Start...")
        number_steps_this_episode = 0
        while not done:
            number_steps_this_episode += 1
            new_state, reward, done, info = env.step(policy_copy[current_state])
            # frozen_lake_env.render()
            # input("Hit Enter...")
            if done:
                # print('Done! State: {0}, Reward: {1}'.format(new_state, reward))
                if reward == 1.0:
                    # print('Got That freesby')
                    success_count += 1
                    number_steps_list.append(number_steps_this_episode)
                else:
                    # print('Help, I am sinking')
                    pass

                # print("Number of steps for episode {0}: {1}".format(episode + 1, number_steps_this_episode))
            current_state = new_state
        # print('Final state: ', current_state)
        # frozen_lake_env.render()
    mean_number_steps_per_successful_episode = np.mean(np.array(number_steps_list))
    success_rate = float(success_count) / float(number_episodes)
    # print("Success Rate: {0}".format(success_rate))
    return success_rate, mean_number_steps_per_successful_episode

def follow_policy_ai_modern_approach(grid_mdp, policy, number_episodes=10):
    # > > >.
    # ^ None ^.
    # ^ > ^ <
    #
    # {
    #
    #     (0, 1): (0, 1),
    #     (1, 2): (1, 0),
    #     (3, 2): None,
    #     (0, 0): (0, 1),
    #     (3, 0): (-1, 0),
    #     (3, 1): None,
    #     (2, 1): (0, 1),
    #     (2, 0): (0, 1),
    #     (2, 2): (1, 0),
    #     (1, 0): (1, 0),
    #     (0, 2): (1, 0)
    #
    # }

    success_count = 0
    number_steps_list = []
    for episode in range(number_episodes):
        done = False
        number_steps_this_episode = 0
        current_state = (0, 0)

        while not done:
            number_steps_this_episode += 1
            action_for_current_state = policy[current_state]
            new_state = grid_mdp.go(current_state, action_for_current_state)
            reward_for_new_state = grid_mdp.reward[new_state]

            current_state = new_state

            if reward_for_new_state == 1:
                done = True
                success_count += 1
                number_steps_list.append(number_steps_this_episode)
            elif reward_for_new_state == -1:
                done = True


    mean_number_steps_per_successful_episode = np.mean(np.array(number_steps_list))
    success_rate = float(success_count) / float(number_episodes)
    # print("Success Rate: {0}".format(success_rate))
    return success_rate, mean_number_steps_per_successful_episode


def print_policy(pie, env):

    policy_to_be_printed = pie.copy()
    policy_to_be_printed = np.reshape(policy_to_be_printed, env.desc.shape)

    # mark all terminal states as -1
    policy_to_be_printed[np.where(env.desc == b'G')] = -1
    policy_to_be_printed[np.where(env.desc == b'H')] = -1

    integer_action_map = {k: v for k, v in enumerate(('LEFT ', 'DOWN ', 'RIGHT', 'UP   '))}
    integer_action_map[-1] = '     '

    separator_line = '-' * ((policy_to_be_printed.shape[1] * 15) + 1)
    print(separator_line)
    for row_index, row in enumerate(policy_to_be_printed):
        for column_index, action in enumerate(row):
            policy_index = row_index * row.shape[0] + column_index
            print("| ", str(policy_index).zfill(2), '-', integer_action_map[action], ' ', end='')
            if column_index == len(row) - 1:
                print(' |')
                print(separator_line)
    #
    # print('----------------------------------------------------------')
    # for row_index, row in enumerate(policy_to_be_printed):
    #     for column_index, action in enumerate(row):
    #         print("| ", env.desc[row_index, column_index].tostring().decode('utf-8'), '-', integer_action_map[action],
    #               ' ', end='')
    #         if column_index == 3:
    #             print(' |')
    #             print('----------------------------------------------------------')


def print_V(V, shape):

    V_to_be_printed = np.reshape(V, shape)

    print("V values: ")
    separator_line = '-' * ((V_to_be_printed.shape[1] * 10) + 1)
    print(separator_line)
    for row_index, row in enumerate(V_to_be_printed):
        for column_index, action in enumerate(row):
            print("| ", np.format_float_positional(V_to_be_printed[row_index][column_index], precision=3, pad_right=3), ' ', end='')
            if column_index == len(row) - 1:
                print(' |')
                print(separator_line)



def print_V_ai_modern_approach(values_dictionary, shape):

    v_matrix = np.zeros(shape, dtype=np.float64)

    for k, v in values_dictionary.items():
        column_index = k[0]
        row_index = k[1]
        v_matrix[row_index, column_index] = v

    v_matrix = np.flip(v_matrix, 0)

    print_V(v_matrix, shape)