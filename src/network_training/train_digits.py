import pickle

# load the training data from the file
with open('data/digits.pkl', 'rb') as file:
    data = pickle.load(file)
training_data = data['training']

# import the class MultilayerPerceptron from the neural_network.py file
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from neural_network import MultilayerPerceptron

# train the network
network = MultilayerPerceptron([784, 16, 16, 10])
network.train(training_data, 10, 30, 3)
network.save_model('digits')