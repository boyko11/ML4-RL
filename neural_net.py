from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np


class NeuralNet(object):

    def __init__(self, num_inputs=8, num_hidden_layer_nodes=16, num_outputs=4, alpha=0.001):

        self.num_inputs = num_inputs
        self.num_hidden_layer_nodes = num_hidden_layer_nodes
        self.num_outputs = num_outputs
        self.alpha = alpha

        self.model = Sequential()
        self.model.add(Dense(num_hidden_layer_nodes, input_dim=num_inputs, activation='relu'))
        #no activation on the output layer per Miguel's advice
        self.model.add(Dense(num_outputs, activation='linear'))

        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    def predict(self, states_matrix):

        num_rows = 1 if states_matrix.ndim == 1 else states_matrix.shape[0]
        return self.model.predict(np.reshape(states_matrix, [num_rows, self.num_inputs]))

    def fit(self, states_matrix, q_network_output):

        num_rows = 1 if states_matrix.ndim == 1 else states_matrix.shape[0]
        q_network_input = np.reshape(states_matrix, [num_rows, self.num_inputs])

        q_network_output = np.reshape(q_network_output, [num_rows, self.num_outputs])

        self.model.fit(q_network_input, q_network_output, batch_size=1, epochs=1, verbose=0)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)



# test_net = NeuralNet(num_inputs=8, num_hidden_layer_nodes=16, num_outputs=4, alpha=0.001)
#
# prediction = test_net.predict(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
#
# print(prediction)


