import pickle
import gzip
import numpy as np

# loads the MNIST data set
def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as file:
        unpickled = pickle._Unpickler(file)
        unpickled.encoding = 'latin1'
        training_data, validation_data, test_data = unpickled.load()
    return (training_data, validation_data, test_data)

# makes a vector for the output layer from the correct output label
def convert_to_vector(correct_output):
    output_layer = np.zeros((10, 1))
    output_layer[correct_output] = 1
    return output_layer

training_data, _, test_data = load_data()
# convert training data into the right format
training_inputs = [np.reshape(input_data, (784, 1)) for input_data in training_data[0]]
training_outputs = [convert_to_vector(output_data) for output_data in training_data[1]]
training_data = [{
    'input': training_inputs[example],
    'output': training_outputs[example]
} for example in range(len(training_inputs))]
# convert testing data into the right format
test_inputs = [np.reshape(input_data, (784, 1)) for input_data in test_data[0]]
test_outputs = test_data[1]
test_data = [{
    'input': test_inputs[example],
    'output': test_outputs[example]
} for example in range(len(test_inputs))]

# save data to a file
with open(f'data/digits.pkl', 'wb') as file:
    pickle.dump({'training': training_data, 'testing': test_data}, file)