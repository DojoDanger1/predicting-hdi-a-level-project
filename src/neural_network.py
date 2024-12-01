import numpy as np
import copy
import pickle
import random

class MultilayerPerceptron(object):
    # constructor method
    def __init__(self, layer_sizes):
        # initialise layer sizes constant
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes)-1
        # initialise matrices for the different quantities
        self.activations = [np.matrix(np.zeros(shape=(layer_size,1))) for layer_size in layer_sizes]
        self.weights = [np.matrix(np.random.randn(layer_sizes[index+1],layer_size)) for index, layer_size in enumerate(layer_sizes[:-1])]
        self.biases = [np.matrix(np.random.randn(layer_size,1)) for layer_size in layer_sizes]
        self.z = [np.matrix(np.zeros(shape=(layer_size,1))) for layer_size in layer_sizes]
        self.errors = [np.matrix(np.zeros(shape=(layer_size,1))) for layer_size in layer_sizes]
        # initialise lists for training states
        self.training_activations = []
        self.training_z = []
        self.training_errors = []

    # carry out the feedforward algorithm
    def feedforward(self):
        for layer in range(0, self.L):
            self.z[layer+1] = np.matmul(self.weights[layer], self.activations[layer]) + self.biases[layer+1]
            self.activations[layer+1] = sigmoid(self.z[layer+1])
    
    # carry out the backpropogation algorithm
    def backpropogate(self, examples, learning_rate):
        # reset the training lists
        self.training_activations = []
        self.training_z = []
        self.training_errors = []
        # iterate over each example
        for example in examples:
            # reset the activations, z-values and errors
            self.activations = [np.matrix(np.zeros(shape=(layer_size,1))) for layer_size in self.layer_sizes]
            self.z = [np.matrix(np.zeros(shape=(layer_size,1))) for layer_size in self.layer_sizes]
            self.errors = [np.matrix(np.zeros(shape=(layer_size,1))) for layer_size in self.layer_sizes]
            # make a feedforward pass
            self.activations[0] = example['input']
            self.feedforward()
            # calculate the error in the output layer
            self.errors[self.L] = np.multiply(
                self.activations[self.L] - example['output'],
                sigmoid_prime(self.z[self.L])
            )
            # backpropogate the error to previous layers
            for layer in range(self.L, 1, -1):
                self.errors[layer-1] = np.multiply(
                    np.matmul(
                        self.weights[layer-1].T,
                        self.errors[layer]
                    ),
                    sigmoid_prime(self.z[layer-1])
                )
            # save the activations, z-values and errors
            self.training_activations.append(copy.deepcopy(self.activations))
            self.training_z.append(copy.deepcopy(self.z))
            self.training_errors.append(copy.deepcopy(self.errors))
        # use gradient descent to update the weights and biases
        for layer in range(self.L, 0, -1):
            self.weights[layer-1] -= (learning_rate/len(examples)) * sum(
                [np.matmul(
                    self.training_errors[example][layer],
                    self.training_activations[example][layer-1].T
                ) for example in range(len(examples))],
                np.matrix(np.zeros(self.weights[layer-1].shape)) # (extra argument in sum function to add matrices instead of numbers)
            )
            self.biases[layer] -= (learning_rate/len(examples)) * sum(
                [self.training_errors[example][layer] for example in range(len(examples))],
                np.matrix(np.zeros(self.biases[layer].shape)) # (extra argument in sum function to add matrices instead of numbers)
            )
    
    # run the feedforward algorithm on a set of input data and return the result
    def predict(self, input_data):
        self.activations[0] = input_data
        self.feedforward()
        return self.activations[self.L]
    
    # carry out stochastic gradient descent over a number of epochs to train a model
    def train(self, examples, mini_batch_size, num_epochs, learning_rate):
        for epoch in range(1, num_epochs+1):
            print(f'training epoch {epoch}/{num_epochs}...') # log message
            # randomly split into mini batches
            random.shuffle(examples)
            mini_batches = [examples[(batch_number * mini_batch_size):((batch_number+1) * mini_batch_size)] for batch_number in range(int(np.ceil(len(examples) / mini_batch_size)))]
            # backpropogate for each mini batch
            for num, mini_batch in enumerate(mini_batches):
                print(f' mini batch {num}/{len(mini_batches)}...') # log message
                self.backpropogate(mini_batch, learning_rate)
    
    # saves weights and biases to an external file in the models folder
    def save_model(self, filename):
        parameters = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes
        }
        with open(f'models/{filename}.pkl', 'wb') as file:
            pickle.dump(parameters, file)
    
    # loads a model from a file path
    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            parameters = pickle.load(file)
        if self.layer_sizes != parameters['layer_sizes']:
            raise Exception('layer sizes do not match!')
        self.weights = parameters['weights']
        self.biases = parameters['biases']

# activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of the activation function
def sigmoid_prime(x):
    return np.multiply(sigmoid(x), 1-sigmoid(x))