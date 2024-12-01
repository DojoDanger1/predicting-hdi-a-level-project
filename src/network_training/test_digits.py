import pickle

# load the test data from the file
with open('data/digits.pkl', 'rb') as file:
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
network = MultilayerPerceptron([784, 16, 16, 10])
network.load_model('models/digits.pkl')
success = 0
for total_so_far, example in enumerate(testing_data):
    prediction = network.predict(example['input'])
    list_of_probabilities = [prediction.item(x,0) for x in range(10)]
    if list_of_probabilities.index(max(list_of_probabilities)) == example['output']:
        success += 1
    print(f'after {total_so_far+1} examples, the success rate is {(success/(total_so_far+1))*100}%')