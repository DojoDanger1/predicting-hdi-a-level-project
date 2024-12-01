import pickle

# load the training data from the file
with open('data/hdi.pkl', 'rb') as file:
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
network = MultilayerPerceptron([24, 18, 12, 6, 3, 1])
network.train(training_data, 5, 50, 0.2)
network.save_model('hdi-temp')