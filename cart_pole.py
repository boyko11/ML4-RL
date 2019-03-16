import gym
from neural_net import NeuralNet
from agent import Agent
import random
import numpy as np
from collections import deque

#params
N_REPLAY_BUFFER_MAX_LENGTH = 2000
NUMBER_EPISODES = 3000
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MINIBATCH_SIZE = 32
INSTANCE = 1

#NeuralNet params
class NN_Param:

    def __init__(self):
        self.num_hidden_layer_nodes = 32
        self.alpha = 0.001

NN_Params = NN_Param()

env = gym.make('CartPole-v0')


#Initialize action-value function Q with random weights
#This is Keras neural net - Keras takes care of random weights initialization
# q_network = NeuralNet(num_inputs=8,
#                       num_hidden_layer_nodes=NN_Params.num_hidden_layer_nodes, num_outputs=env.action_space.n,
#                       alpha=NN_Params.alpha)
q_network = NeuralNet(num_inputs=4,
                      num_hidden_layer_nodes=NN_Params.num_hidden_layer_nodes, num_outputs=env.action_space.n,
                      alpha=NN_Params.alpha)


agent = Agent(q_network, env.action_space.n, GAMMA, EPSILON, EPSILON_DECAY, N_REPLAY_BUFFER_MAX_LENGTH)

episodes_rewards_queue = deque(maxlen=25)
for episode_number in range(NUMBER_EPISODES):

    done = False
    iter_count = 0
    episode_reward = 0

    current_state = env.reset()
    landed = False
    while not done:
        #With probability epsilon select a random action at a_t
        #Otherwise select greedily with respect to the Q_network output
        this_action = agent.act(current_state)
        if agent.epsilon > EPSILON_MIN:
            agent.epsilon *= agent.epsilon_decay

        #Execute action a_t in emulator and observe reward r_t and new_state
        new_state, reward, done, info = env.step(this_action)

        # Store transition in D
        # agent.store_transition(current_state, this_action, reward, new_state, (reward == -100 or reward == 100))
        # REMEMBER ME
        agent.store_transition(current_state, this_action, reward, new_state, done)
        current_state = new_state

        iter_count += 1
        episode_reward += reward

    episodes_rewards_queue.append(episode_reward)
    avg_reward = np.mean(np.array(list(episodes_rewards_queue))) if len(episodes_rewards_queue) > 0 else episode_reward
    print(
        'Episode {0} Steps: {1}. Reward {2}, step reward: {3}, Avg Reward: {4}, Epsilon {5}'.format(episode_number + 1,
                                                                                                iter_count,
                                                                                                episode_reward,
                                                                                                reward,
                                                                                                avg_reward,
                                                                                                agent.epsilon))
    #Sample random minibatch of transitions
    if len(agent.state_buffer_D) >= MINIBATCH_SIZE:

        random_indices = random.sample(range(len(agent.state_buffer_D)), MINIBATCH_SIZE)
        current_states = np.asarray(agent.state_buffer_D)[random_indices]
        actions = np.asarray(agent.action_buffer_D)[random_indices]
        rewards = np.asarray(agent.reward_buffer_D)[random_indices]
        next_states = np.asarray(agent.next_state_buffer_D)[random_indices]
        is_terminals = np.asarray(agent.is_terminal_buffer_D)[random_indices]

        q_target_action_values_matrix = agent.q_network.predict(next_states)

        q_targets = rewards + agent.gamma * np.amax(q_target_action_values_matrix, axis=1)

        q_targets[is_terminals] = rewards[is_terminals]

        q_predicted_action_values_matrix = agent.q_network.predict(current_states)

        q_predicted_action_values_matrix[range(MINIBATCH_SIZE), actions] = q_targets

        agent.q_network.fit(current_states, q_predicted_action_values_matrix)












print('~~~~~~~~~~~~~~~~~~~~~~')
