import numpy as np
import random as rand
from collections import deque

class Agent(object):

    def __init__(self, q_network, num_actions, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, replay_buffer_size=1000):

        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_network = q_network
        # Initialize replay memory D to capacity N
        self.state_buffer_D = deque(maxlen=replay_buffer_size)
        self.action_buffer_D = deque(maxlen=replay_buffer_size)
        self.reward_buffer_D = deque(maxlen=replay_buffer_size)
        self.next_state_buffer_D = deque(maxlen=replay_buffer_size)
        self.is_terminal_buffer_D = deque(maxlen=replay_buffer_size)

    def act(self, current_state):

        choose_random_action = np.random.multinomial(1, [self.epsilon, (1 - self.epsilon)])[0]

        if choose_random_action:
            action = rand.randint(0, self.num_actions - 1)
        else:
            all_action_values = self.q_network.predict(current_state)
            action = np.argmax(all_action_values[0])

        return action

    def process_action_result(self, state, action, reward, next_state, is_terminal):

        q_target_action_values = self.q_network.predict(next_state)
        q_target = reward + self.gamma * np.max(q_target_action_values)
        if is_terminal:
            q_target = reward

        q_current_action_values = self.q_network.predict(state)
        q_current_action_values[0][action] = q_target
        self.q_network.fit(state, q_current_action_values)

    def store_transition(self, current_state, current_action, reward, new_state, is_terminal):

        self.state_buffer_D.append(current_state)
        self.action_buffer_D.append(current_action)
        self.reward_buffer_D.append(reward)
        self.next_state_buffer_D.append(new_state)
        self.is_terminal_buffer_D.append(is_terminal)




