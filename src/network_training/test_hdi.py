import pickle

# load the test data from the file
with open('data/hdi.pkl', 'rb') as file:
    data = pickle.load(file)
testing_data = data['testing']

# import the class MultilayerPerceptron from the neural_network.py file
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from neural_network import MultilayerPerceptron

# iterate over each of the testing examples and see how many it gets right
network = MultilayerPerceptron([24, 18, 12, 6, 3, 1])
network.load_model('models/hdi-temp.pkl')
differences = []
for num, example in enumerate(testing_data):
    correctAnswer = example["output"].item(0,0)
    prediction = round(network.predict(example['input']).item(0,0), 3)
    difference = round(abs(correctAnswer-prediction), 3)
    print(f'({num}) correct output: {correctAnswer}, prediction: {prediction}, difference: {difference}')
    differences.append(difference)

import numpy as np
print(f'average difference: {np.mean(differences)}')